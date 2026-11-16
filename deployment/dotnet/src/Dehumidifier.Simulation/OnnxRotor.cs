using System.Text.Json;
using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Dehumidifier.Simulation;

internal sealed class OnnxRotor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly double[] _inputScale;
    private readonly double[] _inputMin;
    private readonly double[] _outputScale;
    private readonly double[] _outputMin;
    private readonly bool[] _log1pOutputs;
    private readonly int _inputCount;
    private readonly int _outputCount;
    private bool _disposed;

    public string Profile { get; }
    public IReadOnlyList<string> InputColumns { get; }
    public IReadOnlyList<string> OutputColumns { get; }

    public OnnxRotor(string modelDirectory, string profile)
    {
        InferenceSession? session = null;
        try
        {
            if (profile is not ("xf" or "main")) throw new SimulationException("MODEL_CONTRACT_INVALID", "转轮profile只能为xf或main");
            Profile = profile;
            var root = Path.GetFullPath(modelDirectory);
            var contractPath = Path.Combine(root, "model_contract.json");
            var preprocessingPath = Path.Combine(root, "preprocessing.json");
            if (!File.Exists(contractPath) || !File.Exists(preprocessingPath)) throw new SimulationException("MODEL_CONTRACT_MISSING", $"模型目录缺少契约文件: {root}");
            using var contractDoc = JsonDocument.Parse(File.ReadAllText(contractPath));
            var contract = contractDoc.RootElement;
            if (!contract.TryGetProperty("inference_numeric_policy", out var precision) || precision.GetString() != "float64_body_float32_io_v1")
                throw new SimulationException("MODEL_CONTRACT_INVALID", "需要双精度内部推理的配套模型，请整套更新dll和models");
            if (!contract.TryGetProperty("onnx_session_policy", out var sessionPolicy) || sessionPolicy.GetString() != "cpu_sequential_no_fusion_no_prepacking_v1")
                throw new SimulationException("MODEL_CONTRACT_INVALID", "模型推理会话数值策略不匹配");
            if (contract.GetProperty("schema_version").GetInt32() != 2 || contract.GetProperty("preprocessing_file").GetString() != "preprocessing.json")
                throw new SimulationException("MODEL_CONTRACT_INVALID", "需要本次导出的schema2部署契约，请勿混用旧models目录");
            var models = contract.GetProperty("models").EnumerateArray().ToArray();
            if (models.Length != 2 || !models.Select(x => x.GetProperty("profile").GetString()).Order().SequenceEqual(new[] { "main", "xf" }))
                throw new SimulationException("MODEL_CONTRACT_INVALID", "部署契约必须各包含一个xf和main模型");
            VerifyHash(preprocessingPath, contract.GetProperty("preprocessing_sha256").GetString());
            var model = models.Single(x => x.GetProperty("profile").GetString() == profile);
            if (model.ValueKind == JsonValueKind.Undefined) throw new SimulationException("MODEL_CONTRACT_INVALID", $"模型契约缺少 profile={profile}");
            _inputName = model.GetProperty("input_name").GetString() ?? "input";
            _outputName = model.GetProperty("output_name").GetString() ?? "output";
            InputColumns = model.GetProperty("input_cols").EnumerateArray().Select(x => x.GetString()!).ToArray();
            OutputColumns = model.GetProperty("direct_outputs").EnumerateArray().Select(x => x.GetString()!).ToArray();
            _inputCount = InputColumns.Count; _outputCount = OutputColumns.Count;
            if (_outputCount != 6 || _inputCount != 9) throw new SimulationException("MODEL_CONTRACT_INVALID", $"{profile} 模型输入/输出维度不符合固定契约");
            if (!model.TryGetProperty("output_constraint", out var constraint) || constraint.GetProperty("type").GetString() != "humidity_softplus_lower_bound" || constraint.GetProperty("beta").GetDouble() != 50) throw new SimulationException("MODEL_CONTRACT_INVALID", $"{profile} 基线要求图内Softplus湿度非负头(beta=50)");
            ValidateColumns(profile, InputColumns, OutputColumns);
            using var prepDoc = JsonDocument.Parse(File.ReadAllText(preprocessingPath));
            if (prepDoc.RootElement.GetProperty("schema_version").GetInt32() != 2)
                throw new SimulationException("MODEL_CONTRACT_INVALID", "预处理schema版本错误");
            var prep = prepDoc.RootElement.GetProperty("profiles").GetProperty(profile);
            var profilePath = Path.Combine(root, $"preprocessing_{profile}.json");
            VerifyHash(profilePath, model.GetProperty("preprocessing_profile_sha256").GetString());
            using var profileDoc = JsonDocument.Parse(File.ReadAllText(profilePath));
            if (JsonSerializer.Serialize(prep) != JsonSerializer.Serialize(profileDoc.RootElement)
                || !prep.GetProperty("input_columns").EnumerateArray().Select(x => x.GetString()).SequenceEqual(InputColumns)
                || !prep.GetProperty("output_columns").EnumerateArray().Select(x => x.GetString()).SequenceEqual(OutputColumns)
                || prep.GetProperty("clip").GetBoolean()
                || !prep.GetProperty("feature_range").EnumerateArray().Select(x => x.GetDouble()).SequenceEqual(new[] { -1.0, 1.0 })
                || prep.GetProperty("input_scaling").GetString() != "sklearn_float32_inplace_float64_coefficients_v1")
                throw new SimulationException("MODEL_CONTRACT_INVALID", $"{profile} 预处理列、缩放或分文件与汇总文件不一致");
            _inputScale = Numbers(prep, "input_scale", _inputCount); _inputMin = Numbers(prep, "input_min", _inputCount);
            _outputScale = Numbers(prep, "output_scale", _outputCount); _outputMin = Numbers(prep, "output_min", _outputCount);
            if (_inputScale.Concat(_outputScale).Any(x => x <= 0)) throw new SimulationException("MODEL_CONTRACT_INVALID", "MinMax缩放系数必须为正");
            if (!prep.GetProperty("log1p_output_columns").EnumerateArray().Select(x => x.GetString()).SequenceEqual(new[] { "处理出口含湿量" }))
                throw new SimulationException("MODEL_CONTRACT_INVALID", "仅处理出口含湿量允许log1p反变换");
            _log1pOutputs = Enumerable.Repeat(false, _outputCount).ToArray();
            foreach (var item in prep.GetProperty("log1p_output_columns").EnumerateArray())
            { var name = item.GetString(); var index = Array.IndexOf(OutputColumns.ToArray(), name!); if (index < 0) throw new SimulationException("MODEL_CONTRACT_INVALID", $"{profile} log1p 列不在输出契约中: {name}"); _log1pOutputs[index] = true; }
            if (model.GetProperty("onnx_file").GetString() != profile + ".onnx") throw new SimulationException("MODEL_CONTRACT_INVALID", "模型文件名与profile不一致");
            var modelFile = Path.Combine(root, profile + ".onnx");
            if (!File.Exists(modelFile)) throw new SimulationException("MODEL_MISSING", $"模型文件不存在: {modelFile}");
            VerifyHash(modelFile, model.GetProperty("onnx_sha256").GetString());
            // ORT 1.20 may fuse double SiLU to a float-only QuickGelu kernel.
            // Lock the validated CPU arithmetic path instead of implicit fusions.
            using var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
            };
            options.AddSessionConfigEntry("session.disable_prepacking", "1");
            session = new InferenceSession(modelFile, options);
            if (session.InputMetadata.Count != 1 || session.OutputMetadata.Count != 1
                || !session.InputMetadata.TryGetValue(_inputName, out var input) || !session.OutputMetadata.TryGetValue(_outputName, out var output))
                throw new SimulationException("MODEL_CONTRACT_INVALID", $"ONNX 张量名不匹配: {_inputName}/{_outputName}");
            ValidateTensor(input, _inputCount); ValidateTensor(output, _outputCount);
            _session = session;
        }
        catch (SimulationException) { session?.Dispose(); throw; }
        catch (Exception ex) { session?.Dispose(); throw new SimulationException("MODEL_LOAD_FAILED", $"加载 {profile} ONNX 模型失败: {ex.Message}", ex); }
    }

    public double[] Predict(double[] physicalInput)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (physicalInput.Length != _inputCount) throw new SimulationException("MODEL_INPUT_INVALID", $"{Profile} 输入长度必须为 {_inputCount}");
        var scaled = new float[_inputCount];
        for (var i = 0; i < _inputCount; i++)
        {
            if (!double.IsFinite(physicalInput[i])) throw new SimulationException("MODEL_INPUT_INVALID", $"{Profile} 输入存在非有限值");
            var value = (float)physicalInput[i];
            var product = (float)((double)value * _inputScale[i]);
            scaled[i] = (float)((double)product + _inputMin[i]);
            if (!float.IsFinite(scaled[i])) throw new SimulationException("MODEL_INPUT_INVALID", $"{Profile} 输入缩放超出float32范围");
        }
        var tensor = new DenseTensor<float>(scaled, new[] { 1, _inputCount });
        var input = NamedOnnxValue.CreateFromTensor(_inputName, tensor);
        using var outputs = _session.Run(new[] { input });
        var tensorOutput = outputs.FirstOrDefault(x => x.Name == _outputName)?.AsTensor<float>() ?? throw new SimulationException("MODEL_INFERENCE_FAILED", $"{Profile} ONNX 未返回 {_outputName}");
        if (tensorOutput.Length != _outputCount) throw new SimulationException("MODEL_INFERENCE_FAILED", $"{Profile} ONNX 输出长度异常: {tensorOutput.Length}");
        var physical = new double[_outputCount];
        for (var i = 0; i < _outputCount; i++)
        {
            var y = tensorOutput.GetValue(i);
            var restored = (double)y - _outputMin[i];
            restored /= _outputScale[i];
            if (_log1pOutputs[i]) restored = Expm1(restored);
            if (!double.IsFinite(restored)) throw new SimulationException("MODEL_INFERENCE_FAILED", $"{Profile} 逆变换产生非有限值");
            physical[i] = restored;
        }
        return physical;
    }

    private static double Expm1(double value)
    {
        if (Math.Abs(value) > 1e-5) return Math.Exp(value) - 1.0;
        var value2 = value * value;
        return value + value2 / 2.0 + value2 * value / 6.0 + value2 * value2 / 24.0;
    }

    private static double[] Numbers(JsonElement parent, string name, int count)
    {
        var values = parent.GetProperty(name).EnumerateArray().Select(x => x.GetDouble()).ToArray();
        if (values.Length != count || values.Any(x => !double.IsFinite(x))) throw new SimulationException("MODEL_CONTRACT_INVALID", $"预处理参数 {name} 维度或数值无效");
        return values;
    }
    private static void VerifyHash(string path, string? expected)
    {
        if (!File.Exists(path)) throw new SimulationException("MODEL_MISSING", $"模型配套文件不存在: {path}");
        var actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));
        if (expected is null || !actual.Equals(expected, StringComparison.OrdinalIgnoreCase))
            throw new SimulationException("MODEL_CONTRACT_INVALID", $"模型配套文件SHA256不匹配，请整套更新models: {Path.GetFileName(path)}");
    }
    private static void ValidateTensor(NodeMetadata metadata, int columns)
    {
        if (metadata.ElementType != typeof(float) || metadata.Dimensions.Length != 2
            || metadata.Dimensions[1] != columns || metadata.Dimensions[0] is not (-1 or 1))
            throw new SimulationException("MODEL_CONTRACT_INVALID", $"ONNX张量必须为float32[batch,{columns}]");
    }
    private static void ValidateColumns(string profile, IReadOnlyList<string> inputs, IReadOnlyList<string> outputs)
    {
        string[] xf = { "转轮厚度", "转轮直径", "处理入口温度(℃)", "入口含湿量(g/kg)", "处理风量", "冷吹风量", "再生温度(℃)", "再生风量", "再生入口含湿量" };
        string[] main = { "转轮厚度", "转轮直径", "入口含湿量(g/kg)", "转轮转速(rpm)", "再生温度(℃)", "冷吹入口温度(℃)", "处理入口温度(℃)", "处理风量", "冷吹风量" };
        string[] expected = { "处理出口温度", "处理出口含湿量", "冷吹出口温度", "冷吹出口含湿量", "再生出口温度", "再生出口含湿量" };
        if (!inputs.SequenceEqual(profile == "xf" ? xf : main) || !outputs.SequenceEqual(expected)) throw new SimulationException("MODEL_CONTRACT_INVALID", $"{profile} 输入/输出列顺序不符合固定契约");
    }

    public void Dispose()
    { if (!_disposed) { _session.Dispose(); _disposed = true; } }
}
