// Audit trail: every decision, approval, refusal and backend attempt.
// Append-only; the JSONL file sink mirrors the Python decisions.jsonl
// habit (bounded single backup, oldest dropped).

using System.Text.Json;

namespace CyberMarl.Deployment.Core;

public sealed record AuditEntry(
    DateTimeOffset Timestamp,
    string Event,      // decided | queued | approved | denied | executed | failed | blocked | rebind
    string Operation,
    string Target,     // real asset identity (never a bare CC4 slot)
    string Mode,
    bool Applied,
    string Details,
    string Actor);

public interface IAuditSink
{
    void Record(AuditEntry entry);
    IReadOnlyList<AuditEntry> Entries { get; }
}

public sealed class InMemoryAuditLog : IAuditSink
{
    private readonly List<AuditEntry> _entries = new();
    private readonly object _gate = new();

    public void Record(AuditEntry entry)
    {
        lock (_gate) { _entries.Add(entry); }
    }

    public IReadOnlyList<AuditEntry> Entries
    {
        get { lock (_gate) { return _entries.ToList(); } }
    }
}

/// <summary>Append-only JSONL file sink (50 MiB rotation, one backup).</summary>
public sealed class JsonlAuditSink : IAuditSink, IDisposable
{
    public const long DefaultMaxBytes = 50L * 1024 * 1024;

    private readonly InMemoryAuditLog _memory = new();
    private readonly string _path;
    private readonly long _maxBytes;
    private StreamWriter? _writer;
    private readonly object _gate = new();
    private bool _disposed;

    public JsonlAuditSink(string path, long maxBytes = DefaultMaxBytes)
    {
        _path = path;
        _maxBytes = maxBytes;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".");
        _writer = new StreamWriter(new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read))
        {
            AutoFlush = true
        };
    }

    public void Record(AuditEntry entry)
    {
        _memory.Record(entry);
        lock (_gate)
        {
            if (_disposed || _writer is null) return;
            MaybeRotateLocked();
            _writer.WriteLine(JsonSerializer.Serialize(entry));
        }
    }

    public IReadOnlyList<AuditEntry> Entries => _memory.Entries;

    private void MaybeRotateLocked()
    {
        if (_maxBytes <= 0) return;
        try
        {
            if (new FileInfo(_path).Length < _maxBytes) return;
            _writer?.Dispose();
            var backup = _path + ".1";
            if (File.Exists(backup)) File.Delete(backup);
            File.Move(_path, backup);
            _writer = new StreamWriter(new FileStream(_path, FileMode.Append, FileAccess.Write, FileShare.Read))
            {
                AutoFlush = true
            };
        }
        catch { /* rotation best-effort; logging continues */ }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _disposed = true;
            _writer?.Dispose();
            _writer = null;
        }
    }
}
