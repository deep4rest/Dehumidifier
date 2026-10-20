using Dehumidifier.Simulation;

namespace Dehumidifier.WinFormsDemo;

public sealed class MainForm : Form
{
    private readonly Button _run = new() { Text = "开始计算", Width = 110 };
    private readonly Button _cancel = new() { Text = "取消", Width = 90, Enabled = false };
    private readonly TextBox _output = new() { Multiline = true, ReadOnly = true, Dock = DockStyle.Fill, ScrollBars = ScrollBars.Vertical };
    private CancellationTokenSource? _cancellation;
    private Simulator? _simulator;

    public MainForm()
    {
        Text = "双转轮除湿机 C# + ONNX 示例"; Width = 1000; Height = 700;
        var bar = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 48, Padding = new Padding(8) };
        bar.Controls.Add(_run); bar.Controls.Add(_cancel);
        Controls.Add(_output); Controls.Add(bar);
        _run.Click += async (_, _) => await RunAsync();
        _cancel.Click += (_, _) => _cancellation?.Cancel();
        FormClosing += (_, _) => { _cancellation?.Cancel(); _simulator?.Dispose(); };
    }

    private async Task RunAsync()
    {
        _run.Enabled = false; _cancel.Enabled = true; _output.Text = "正在计算……";
        _cancellation = new CancellationTokenSource();
        try
        {
            var baseDirectory = AppContext.BaseDirectory;
            var packagedModels = Path.Combine(baseDirectory, "Dehumidifier", "models");
            var directBuildModels = Path.Combine(baseDirectory, "models");
            _simulator ??= new Simulator(Directory.Exists(packagedModels) ? packagedModels : directBuildModels, Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Dehumidifier", "results", "simulation"));
            var configPath = Path.Combine(baseDirectory, "examples", "qy1070_case.json");
            var config = Simulator.LoadConfiguration(configPath);
            var result = await _simulator.SimulateAsync(config, cancellationToken: _cancellation.Token);
            _output.Text = $"状态：{(result.Converged ? "已收敛" : "未收敛（仅供调试）")}\r\n迭代：{result.Iterations}\r\n送风温度：{result.Supply.TemperatureC:F3} ℃\r\n送风含湿量：{result.Supply.HumidityGPerKg:F5} g/kg\r\n冷负荷：{result.CoolingLoadKW:F5} kW\r\n热负荷：{result.HeatingLoadKW:F5} kW\r\nJSON：{result.ResultJsonPath}\r\nPNG：{result.DashboardPngPath}";
        }
        catch (OperationCanceledException) { _output.Text = "计算已取消。"; }
        catch (SimulationException ex) { _output.Text = $"[{ex.Code}] {ex.Message}"; }
        finally { _cancellation.Dispose(); _cancellation = null; _run.Enabled = true; _cancel.Enabled = false; }
    }
}
