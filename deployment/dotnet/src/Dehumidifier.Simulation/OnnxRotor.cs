using System.Text.Json;
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
        try
        {
            Profile = profile;
            var root = Path.GetFullPath(modelDirectory);
            var contractPath = Path.Combine(root, "model_contract.json");
            var preprocessingPath = Path.Combine(root, "preprocessing.json");
            if (!File.Exists(contractPath) || !File.Exists(preprocessingPath)) throw new SimulationException("MODEL_CONTRACT_MISSING", $"模型目录缺少契约文件: {root}");
            using var contractDoc = JsonDocument.Parse(File.ReadAllText(contractPath));
            var model = contractDoc.RootElement.GetProperty("models").EnumerateArray().FirstOrDefault(x => x.GetProperty("profile").GetString() == profile);
            if (model.ValueKind == JsonValueKind.Undefined) throw new SimulationException("MODEL_CONTRACT_INVALID", $"模型契约缺少 profile={profile}");
            _inputName = model.GetProperty("input_name").GetString() ?? "input";
            _outputName = model.GetProperty("output_name").GetString() ?? "output";
            InputColumns = model.GetProperty("input_cols").EnumerateArray().Select(x => x.GetString()!).ToArray();
            OutputColumns = model.GetProperty("direct_outputs").EnumerateArray().Select(x => x.GetString()!).ToArray();
            _inputCount = InputColumns.Count; _outputCount = OutputColumns.Count;
            if (_outputCount != 6 || (profile == "xf" && _inputCount != 10) || (profile == "main" && _inputCount != 9)) throw new SimulationException("MODEL_CONTRACT_INVALID", $"{profile} 模型输入/输出维度不符合固定契约");
            ValidateColumns(profile, InputColumns, OutputColumns);
            using var prepDoc = JsonDocument.Parse(File.ReadAllText(preprocessingPath));
            var prep = prepDoc.RootElement.GetProperty("profiles").GetProperty(profile);
            _inputScale = Numbers(prep, "input_scale", _inputCount); _inputMin = Numbers(prep, "input_min", _inputCount);
            _outputScale = Numbers(prep, "output_scale", _outputCount); _outputMin = Numbers(prep, "output_min", _outputCount);
            _log1pOutputs = Enumerable.Repeat(false, _outputCount).ToArray();
            foreach (var item in prep.GetProperty("log1p_output_columns").EnumerateArray())
            { var name = item.GetString(); var index = Array.IndexOf(OutputColumns.ToArray(), name!); if (index < 0) throw new SimulationException("MODEL_CONTRACT_INVALID", $"{profile} log1p 列不在输出契约中: {name}"); _log1pOutputs[index] = true; }
            var modelFile = Path.Combine(root, model.GetProperty("onnx_file").GetString()!);
            if (!File.Exists(modelFile)) throw new SimulationException("MODEL_MISSING", $"模型文件不存在: {modelFile}");
            _session = new InferenceSession(modelFile);
            if (!_session.InputMetadata.ContainsKey(_inputName) || !_session.OutputMetadata.ContainsKey(_outputName)) throw new SimulationException("MODEL_CONTRACT_INVALID", $"ONNX 张量名不匹配: {_inputName}/{_outputName}");
        }
        catch (SimulationException) { throw; }
        catch (Exception ex) { throw new SimulationException("MODEL_LOAD_FAILED", $"加载 {profile} ONNX 模型失败: {ex.Message}", ex); }
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
            scaled[i] = value * (float)_inputScale[i] + (float)_inputMin[i];
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
    private static void ValidateColumns(string profile, IReadOnlyList<string> inputs, IReadOnlyList<string> outputs)
    {
        string[] xf = { "转轮厚度", "转轮直径", "处理入口温度(℃)", "入口含湿量(g/kg)", "处理风量", "冷吹风量", "转轮转速(rpm)", "再生温度(℃)", "再生风量", "再生入口含湿量" };
        string[] main = { "转轮厚度", "转轮直径", "入口含湿量(g/kg)", "转轮转速(rpm)", "再生温度(℃)", "冷吹入口温度(℃)", "处理入口温度(℃)", "处理风量", "冷吹风量" };
        string[] expected = { "处理出口温度", "处理出口含湿量", "冷吹出口温度", "冷吹出口含湿量", "再生出口温度", "再生出口含湿量" };
        if (!inputs.SequenceEqual(profile == "xf" ? xf : main) || !outputs.SequenceEqual(expected)) throw new SimulationException("MODEL_CONTRACT_INVALID", $"{profile} 输入/输出列顺序不符合固定契约");
    }

    public void Dispose()
    { if (!_disposed) { _session.Dispose(); _disposed = true; } }
}
