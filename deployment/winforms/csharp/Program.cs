using System.Text.Json;

namespace DehumidifierDemo;

internal static class Program
{
    [STAThread]
    private static void Main(string[] args)
    {
        ApplicationConfiguration.Initialize();
        if (args.Length != 1 || !File.Exists(args[0]))
        {
            MessageBox.Show("请传入 DehumidifierEngine.exe 的完整路径。", "启动参数");
            return;
        }
        Application.Run(new DemoForm(Path.GetFullPath(args[0])));
    }
}

internal sealed class DemoForm : Form
{
    private readonly string _executable;
    private readonly Button _run = new() { Text = "选择工况并仿真", AutoSize = true };
    private readonly Button _cancel = new() { Text = "取消", Enabled = false };
    private readonly TextBox _output = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Both, Dock = DockStyle.Fill };
    private DehumidifierClient? _client;
    private CancellationTokenSource? _cancellation;
    private bool _closing;

    public DemoForm(string executable)
    {
        _executable = executable;
        Text = "除湿机整机稳态仿真接入示例"; Width = 1000; Height = 750;
        var panel = new FlowLayoutPanel { Dock = DockStyle.Top, AutoSize = true };
        panel.Controls.AddRange(new Control[] { _run, _cancel });
        Controls.Add(_output); Controls.Add(panel);
        _run.Click += RunAsync;
        _cancel.Click += (_, _) => _cancellation?.Cancel();
        FormClosing += ClosingAsync;
    }

    private async void RunAsync(object? sender, EventArgs e)
    {
        using var dialog = new OpenFileDialog { Filter = "请求示例 (*.request.json)|*.request.json|JSON (*.json)|*.json" };
        if (dialog.ShowDialog() != DialogResult.OK) return;
        _run.Enabled = false; _cancel.Enabled = true;
        _cancellation = new CancellationTokenSource();
        try
        {
            string text = await File.ReadAllTextAsync(dialog.FileName);
            using var request = JsonDocument.Parse(text);
            var config = request.RootElement.GetProperty("config").Clone();
            string results = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Dehumidifier", "results", "simulation");
            _client ??= new DehumidifierClient(_executable, Array.Empty<string>(), results);
            _output.Text = "正在计算…";
            var response = await _client.SimulateAsync(config, _cancellation.Token);
            _output.Text = response.GetProperty("message").GetString() + Environment.NewLine
                + JsonSerializer.Serialize(response, new JsonSerializerOptions { WriteIndented = true });
        }
        catch (Exception ex)
        {
            _output.Text = ex is OperationCanceledException ? "已取消或计算超时。下次运行会重新启动引擎。" : ex.Message;
            if (_client is not null) { await _client.DisposeAsync(); _client = null; }
        }
        finally
        {
            _cancellation.Dispose(); _cancellation = null;
            _run.Enabled = true; _cancel.Enabled = false;
            if (_closing) Close();
        }
    }

    private async void ClosingAsync(object? sender, FormClosingEventArgs e)
    {
        if (_cancellation is not null)
        {
            e.Cancel = true; _closing = true; _cancellation.Cancel(); return;
        }
        if (_client is not null)
        {
            e.Cancel = true;
            var client = _client; _client = null;
            await client.DisposeAsync(); Close();
        }
    }
}
