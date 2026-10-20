using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace DehumidifierDemo;

/// <summary>一个客户端对应一个串行计算进程。取消/超时后必须创建新客户端。</summary>
public sealed class DehumidifierClient : IAsyncDisposable
{
    private readonly Process _process;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly StringBuilder _logs = new();
    private readonly Task _stderrTask;
    private bool _stopped;

    // EXE 模式 prefixArguments 传空数组；源码模式传 engine.py 的绝对路径。
    public DehumidifierClient(string executable, IEnumerable<string> prefixArguments, string resultsDirectory)
    {
        var start = new ProcessStartInfo(Path.GetFullPath(executable))
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
            StandardInputEncoding = new UTF8Encoding(false),
            StandardOutputEncoding = new UTF8Encoding(false),
            StandardErrorEncoding = new UTF8Encoding(false)
        };
        foreach (var argument in prefixArguments) start.ArgumentList.Add(argument);
        start.ArgumentList.Add("--results-dir");
        start.ArgumentList.Add(Path.GetFullPath(resultsDirectory));
        start.Environment["PYTHONUTF8"] = "1";
        _process = Process.Start(start) ?? throw new InvalidOperationException("计算引擎启动失败");
        _stderrTask = DrainErrorsAsync();
    }

    public Task<JsonElement> SimulateAsync(JsonElement config, CancellationToken cancellationToken = default)
    {
        string id = Guid.NewGuid().ToString("N");
        return SendAsync(new { protocol_version = 1, id, command = "simulate", config }, id, cancellationToken);
    }

    public Task<JsonElement> PingAsync(CancellationToken cancellationToken = default)
    {
        string id = Guid.NewGuid().ToString("N");
        return SendAsync(new { protocol_version = 1, id, command = "ping" }, id, cancellationToken);
    }

    private async Task<JsonElement> SendAsync(object request, string id, CancellationToken token)
    {
        await _gate.WaitAsync(token).ConfigureAwait(false);
        try
        {
            if (_stopped) throw new InvalidOperationException("引擎已停止，请创建新客户端");
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(token);
            deadline.CancelAfter(TimeSpan.FromMinutes(3));
            try
            {
                string json = JsonSerializer.Serialize(request);
                await _process.StandardInput.WriteLineAsync(json.AsMemory(), deadline.Token).ConfigureAwait(false);
                await _process.StandardInput.FlushAsync(deadline.Token).ConfigureAwait(false);
                string? line = await _process.StandardOutput.ReadLineAsync(deadline.Token).ConfigureAwait(false);
                if (line is null) throw new IOException("计算引擎意外退出：" + GetLogs());
                using var document = JsonDocument.Parse(line);
                var response = document.RootElement;
                if (response.GetProperty("protocol_version").GetInt32() != 1 || response.GetProperty("id").GetString() != id)
                    throw new IOException("响应协议或请求编号不匹配");
                // error 和 unconverged 是协议结果，交给界面明确展示，不能当作成功隐藏。
                return response.Clone();
            }
            catch
            {
                Stop(); // 不继续读取取消前残留的响应。
                throw;
            }
        }
        finally { _gate.Release(); }
    }

    private async Task DrainErrorsAsync()
    {
        try
        {
            while (await _process.StandardError.ReadLineAsync().ConfigureAwait(false) is { } line)
                lock (_logs)
                {
                    _logs.AppendLine(line);
                    if (_logs.Length > 16384) _logs.Remove(0, _logs.Length - 8192);
                }
        }
        catch (ObjectDisposedException) { }
        catch (IOException) { }
    }

    public string GetLogs() { lock (_logs) return _logs.ToString(); }

    private void Stop()
    {
        _stopped = true;
        try { if (!_process.HasExited) _process.Kill(entireProcessTree: true); }
        catch (InvalidOperationException) { }
    }

    public async ValueTask DisposeAsync()
    {
        // 宿主应先取消并等待当前请求结束，再释放客户端。
        await _gate.WaitAsync().ConfigureAwait(false);
        try
        {
            Stop();
            await _process.WaitForExitAsync().ConfigureAwait(false);
            await _stderrTask.ConfigureAwait(false);
            _process.Dispose();
        }
        finally { _gate.Release(); }
    }
}
