using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using CyberMarl.Deployment.Core;

namespace CyberMarl.Deployment.App;

public abstract class ViewModelBase : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    protected bool Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return false;
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        return true;
    }

    protected void Raise([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

public sealed class RelayCommand : ICommand
{
    private readonly Func<object?, bool> _canExecute;
    private readonly Action<object?> _execute;

    public RelayCommand(Action<object?> execute, Func<object?, bool>? canExecute = null)
    {
        _execute = execute;
        _canExecute = canExecute ?? (_ => true);
    }

    public RelayCommand(Action<object?> execute, Func<bool> canExecute)
        : this(execute, _ => canExecute())
    {
    }

    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter) => _canExecute(parameter);
    public void Execute(object? parameter) => _execute(parameter);
    public void Refresh() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}

public sealed class AssetRow
{
    public required string AssetId { get; init; }
    public required string Hostname { get; init; }
    public required string Address { get; init; }
    public required string Os { get; init; }
    public required string Segment { get; init; }
    public string Coverage { get; set; } = "unknown";
    public string CoverageDetail { get; set; } = "";
    public string ServicesText { get; set; } = "services unknown (needs telemetry adapter)";
    public string LinksText { get; set; } = "no observed links";
    public string MacsText { get; set; } = "";
    public string? PolicySlot { get; set; } // technical footnote, set after a cycle
}

public sealed class ActionRow : ViewModelBase
{
    private string _status = "proposed";
    public required string Description { get; init; }
    public required string Operation { get; init; }
    public required string Risk { get; init; }
    public bool RequiresApproval { get; init; }
    public bool Blocked { get; init; }
    public string BlockReason { get; init; } = "";
    public int? ApprovalIndex { get; set; }

    public string Status
    {
        get => _status;
        set
        {
            if (Set(ref _status, value))
                StatusChanged?.Invoke(this, EventArgs.Empty);
        }
    }

    public event EventHandler? StatusChanged;
}

public sealed class AuditRow
{
    public required string Time { get; init; }
    public required string Event { get; init; }
    public required string Operation { get; init; }
    public required string Target { get; init; }
    public required string Details { get; init; }
}

public sealed class BindingRow
{
    public required string AssetId { get; init; }
    public required string Agent { get; init; }
    public required string PolicySlot { get; init; }
}

/// <summary>
/// Main SOC view-model. Real topology is the hero; the policy mapping is
/// a technical footnote. All inference/approval calls go to the Python
/// sidecar; this process never loads weights and never touches a network
/// for enforcement.
/// </summary>
public sealed class MainViewModel : ViewModelBase
{
    private static readonly string[] PageTitles =
    {
        "Dashboard", "Real Assets", "Actions",
        "Policy Mapping (technical)", "Audit", "Telemetry & Model",
    };

    private InferenceClient? _client;
    private string _banner = "Welcome. Start with step 1 below: connect to the sidecar.";
    private string _mode = "Shadow";
    private string _health = "not connected";
    private string _contract = "unverified";
    private string _mappingSummary = "no cycle yet";
    private bool _busy;
    private bool _isConnected;
    private bool _sidecarOk;
    private bool _discoveredOk;
    private bool _cycleOk;
    private int _sidebarIndex;
    private int _pendingCount;
    private ActionRow? _selectedAction;

    public MainViewModel()
    {
        Assets = new ObservableCollection<AssetRow>();
        Actions = new ObservableCollection<ActionRow>();
        Actions.CollectionChanged += OnActionsChanged;
        Audit = new ObservableCollection<AuditRow>();
        Bindings = new ObservableCollection<BindingRow>();
        ConnectCommand = new RelayCommand(async _ => await ConnectAsync(), () => !_busy);
        RunCycleCommand = new RelayCommand(async _ => await RunCycleAsync(), () => _client is not null && !_busy);
        ApproveCommand = new RelayCommand(
            async row => await ApproveAsync((ActionRow?)row),
            row => row is ActionRow { RequiresApproval: true, Blocked: false, Status: "queued" } && !_busy);
        RefreshLocalCommand = new RelayCommand(_ => RefreshLocalDiscovery(), () => !_busy);
        NavCommand = new RelayCommand(parameter =>
        {
            if (int.TryParse(parameter?.ToString(), out int index) && index >= 0 && index < PageTitles.Length)
                SidebarIndex = index;
        });
        ModeCommand = new RelayCommand(parameter =>
        {
            var mode = parameter?.ToString();
            if (mode is "Shadow" or "Mock" or "Supervised" or "Live")
                SelectedMode = mode;
        });
        ClearAuditCommand = new RelayCommand(_ => Audit.Clear());
        RefreshDashboard();
    }

    /// <summary>Set by the shell to a confirm prompt. Null = auto-yes.</summary>
    public Func<string, string, Task<bool>>? ConfirmAction { get; set; }

    public ObservableCollection<AssetRow> Assets { get; }
    public ObservableCollection<ActionRow> Actions { get; }
    public ObservableCollection<AuditRow> Audit { get; }
    public ObservableCollection<BindingRow> Bindings { get; }

    public ICommand ConnectCommand { get; }
    public ICommand RunCycleCommand { get; }
    public ICommand ApproveCommand { get; }
    public ICommand RefreshLocalCommand { get; }
    public ICommand NavCommand { get; }
    public ICommand ModeCommand { get; }
    public ICommand ClearAuditCommand { get; }

    public int SidebarIndex
    {
        get => _sidebarIndex;
        set
        {
            if (Set(ref _sidebarIndex, value))
                Raise(nameof(PageTitle));
        }
    }

    public string PageTitle => PageTitles[_sidebarIndex];

    public string Banner { get => _banner; private set => Set(ref _banner, value); }
    public string SelectedMode { get => _mode; set => Set(ref _mode, value); }    public string Health { get => _health; private set => Set(ref _health, value); }
    public string Contract { get => _contract; private set => Set(ref _contract, value); }
    public string MappingSummary { get => _mappingSummary; private set => Set(ref _mappingSummary, value); }
    public bool IsConnected { get => _isConnected; private set => Set(ref _isConnected, value); }
    public int PendingCount { get => _pendingCount; private set => Set(ref _pendingCount, value); }

    public string ActionsNavLabel => PendingCount > 0 ? $"Actions ({PendingCount})" : "Actions";

    public ActionRow? SelectedAction
    {
        get => _selectedAction;
        set
        {
            if (Set(ref _selectedAction, value))
                ((RelayCommand)ApproveCommand).Refresh();
        }
    }

    // Dashboard cards + checklist (plain strings; no converters needed).
    public string ConnectionCard => IsConnected ? "Connected" : "Not connected";
    public string ContractCard => Contract;
    public string AssetsCard => Assets.Count == 0
        ? "none discovered"
        : $"{Assets.Count} machine(s) · {CoveredCount} covered";
    public int CoveredCount => Assets.Count(a =>
        a.Coverage is "covered-local" or "covered-agent");
    public string PendingCard => PendingCount == 0 ? "none" : $"{PendingCount} waiting";
    public string Step1Text => _sidecarOk ? "✓ Step 1 done — sidecar connected, contract verified."
                                          : "Step 1 — Connect to the Python sidecar.";
    public string Step2Text => _discoveredOk ? $"✓ Step 2 done — {Assets.Count} real machine(s) discovered."
                                             : "Step 2 — Discover this machine (shows your real PC, nothing else).";
    public string Step3Text => _cycleOk ? "✓ Step 3 done — policy cycle complete. Review the Actions page."
                                        : "Step 3 — Run a policy cycle (asks the trained model what to do).";

    public void InitializeFromEnvironment(string? url, string? token)
    {
        if (string.IsNullOrWhiteSpace(url) || string.IsNullOrWhiteSpace(token))
        {
            Banner = "Sidecar not configured. Set INFERENCE_URL and INFERENCE_TOKEN, restart, then press Connect. " +
                     "See the Dashboard for the exact commands.";
            return;
        }
        try
        {
            _client?.Dispose();
            _client = new InferenceClient(url, token);
            Banner = $"Sidecar configured at {url}. Step 1: press Connect to verify the frozen contract.";
        }
        catch (InferenceException ex)
        {
            Banner = $"Bad sidecar configuration: {ex.Message}";
        }
        ((RelayCommand)RunCycleCommand).Refresh();
    }

    private async Task ConnectAsync()
    {
        if (_client is null)
        {
            Banner = "No sidecar configured. Set INFERENCE_URL and INFERENCE_TOKEN, restart the app, then Connect.";
            return;
        }
        SetBusy(true);
        try
        {
            var contract = await _client.GetContractAsync();
            var health = await _client.GetHealthAsync();
            Contract = $"obs {contract.ObsDim} / act {contract.ActionDim} / vocab {contract.NumHostTargets}";
            Health = $"{health.Engine} ({health.Mode ?? "?"})";
            IsConnected = true;
            _sidecarOk = true;
            Banner = "Connected and contract verified. Next: step 2 — discover this machine.";
            Record("connected", "-", "sidecar", $"contract {Contract}");
        }
        catch (Exception ex)
        {
            IsConnected = false;
            _sidecarOk = false;
            Banner = $"Connect failed: {ex.Message}. Is the sidecar running? See Dashboard for the command.";
            Record("failed", "connect", "sidecar", ex.Message);
        }
        finally
        {
            SetBusy(false);
            RefreshDashboard();
        }
    }

    private void RefreshLocalDiscovery()
    {
        try
        {
            var (topology, _) = LocalDiscovery.DiscoverThisMachine();
            Assets.Clear();
            foreach (var asset in topology.Assets)
            {
                Assets.Add(new AssetRow
                {
                    AssetId = asset.AssetId,
                    Hostname = asset.Hostname,
                    Address = asset.Ips.Count > 0 ? string.Join(", ", asset.Ips) : "no address",
                    Os = string.IsNullOrWhiteSpace(asset.OsName) ? "unknown OS" : asset.OsName,
                    Segment = asset.SegmentId,
                    Coverage = asset.Coverage,
                    CoverageDetail = "sensed on this machine (processes + TCP observed; services need adapter)",
                    MacsText = asset.Macs.Count > 0 ? string.Join(", ", asset.Macs) : "no MAC observed",
                });
            }
            _discoveredOk = Assets.Count > 0;
            Banner = Assets.Count == 1
                ? "Discovery found your 1 real machine. Nothing else is claimed. Next: step 3 — run a policy cycle."
                : $"Discovery found {Assets.Count} real machine(s). Next: step 3 — run a policy cycle.";
        }
        catch (Exception ex)
        {
            _discoveredOk = false;
            Banner = $"Local discovery failed: {ex.Message}";
        }
        finally { RefreshDashboard(); }
    }

    private async Task RunCycleAsync()
    {
        if (_client is null)
        {
            Banner = "No sidecar configured — Connect first (step 1).";
            return;
        }
        SetBusy(true);
        try
        {
            var (topology, batch) = LocalDiscovery.DiscoverThisMachine();
            var result = await _client.RunCycleAsync(topology, batch);
            MappingSummary = $"mapping {result.Mapping.Digest}: {result.Mapping.Bindings.Count} bound, " +
                             $"{result.Mapping.Unpopulated.Count} unpopulated padding, {result.Mapping.Overflow.Count} overflow";

            // Server-authoritative real topology (coverage, services,
            // links as the cycle SAW them). Policy slots attach as
            // footnotes only -- never infrastructure.
            var realTopology = InferenceClient.ToModel(result.Topology);
            var linksByAsset = realTopology.Links?
                .GroupBy(l => l.AssetId)
                .ToDictionary(g => g.Key,
                    g => string.Join("; ", g.Select(l => $"{l.Peer} ({l.Via})")))
                ?? new Dictionary<string, string>();
            Assets.Clear();
            var slotByAsset = result.Mapping.Bindings.ToDictionary(b => b.AssetId, b => b.PolicySlot);
            foreach (var asset in realTopology.Assets)
            {
                slotByAsset.TryGetValue(asset.AssetId, out var slot);
                linksByAsset.TryGetValue(asset.AssetId, out var links);
                var services = asset.Services;
                Assets.Add(new AssetRow
                {
                    AssetId = asset.AssetId,
                    Hostname = asset.Hostname,
                    Address = asset.Ips.Count > 0 ? string.Join(", ", asset.Ips) : "no address",
                    Os = string.IsNullOrWhiteSpace(asset.OsName) ? "unknown OS" : asset.OsName,
                    Segment = asset.SegmentId,
                    Coverage = asset.Coverage,
                    CoverageDetail = string.IsNullOrWhiteSpace(asset.CoverageDetail)
                        ? "no coverage detail" : asset.CoverageDetail,
                    ServicesText = services is null || services.Count == 0
                        ? "services unknown (needs telemetry adapter)"
                        : $"{services.Count} observed: " + string.Join(", ", services.Take(12)) +
                          (services.Count > 12 ? "…" : ""),
                    LinksText = string.IsNullOrEmpty(links) ? "no observed links" : links,
                    MacsText = asset.Macs.Count > 0 ? string.Join(", ", asset.Macs) : "no MAC observed",
                    PolicySlot = slot is null ? "overflow (inventory)" : $"policy slot (technical): {slot}",
                });
            }
            _discoveredOk = Assets.Count > 0;

            Bindings.Clear();
            foreach (var binding in result.Mapping.Bindings)
            {
                Bindings.Add(new BindingRow
                {
                    AssetId = binding.AssetId,
                    Agent = binding.Agent,
                    PolicySlot = binding.PolicySlot,
                });
            }

            Actions.Clear();
            var queueIndex = 0;
            foreach (var plan in result.RealActions)
            {
                var row = new ActionRow
                {
                    Description = plan.Description,
                    Operation = plan.Operation,
                    Risk = plan.Risk,
                    RequiresApproval = plan.RequiresApproval,
                    Blocked = plan.Blocked,
                    BlockReason = plan.BlockReason,
                    Status = plan.Blocked ? "blocked" : plan.RequiresApproval ? "queued" : "proposed",
                };
                if (!plan.Blocked && plan.RequiresApproval)
                    row.ApprovalIndex = queueIndex++;
                Actions.Add(row);
                Record(plan.Blocked ? "blocked" : "decided",
                    plan.Operation,
                    string.IsNullOrEmpty(plan.Hostname) ? "local" : plan.Hostname,
                    plan.Description);
            }
            _cycleOk = true;
            RefreshPending();

            Banner = PendingCount > 0
                ? $"Cycle done ({SelectedMode}). {PendingCount} action(s) need your approval — open the Actions page."
                : $"Cycle done ({SelectedMode}). No approvals needed. {MappingSummary}.";
        }
        catch (Exception ex)
        {
            Banner = $"Cycle failed: {ex.Message}";
            Record("failed", "cycle", "sidecar", ex.Message);
        }
        finally
        {
            SetBusy(false);
            RefreshDashboard();
        }
    }

    private async Task ApproveAsync(ActionRow? row)
    {
        if (_client is null || row?.ApprovalIndex is null) return;
        if (SelectedMode == "Live")
        {
            bool confirmed = await (ConfirmAction?.Invoke(
                "Live approval — executes for real",
                $"'{row.Description}' will execute for real via the site backend. Approve?")
                ?? Task.FromResult(true));
            if (!confirmed)
            {
                Banner = "Live approval cancelled — nothing executed.";
                return;
            }
        }
        SetBusy(true);
        try
        {
            var outcome = await _client.ApproveAsync(row.ApprovalIndex.Value, Environment.UserName);
            row.Status = outcome.Applied ? "approved" : "approval-failed";
            Record(outcome.Applied ? "approved" : "failed",
                row.Operation, "approval", outcome.Applied ? outcome.Details : outcome.Error);
            Banner = outcome.Applied
                ? $"Approved and applied: {row.Description}."
                : $"Approval recorded but not applied: {outcome.Error}";
            RefreshPending();
        }
        catch (Exception ex)
        {
            row.Status = "approval-failed";
            Banner = $"Approval failed: {ex.Message}";
            Record("failed", row.Operation, "approval", ex.Message);
        }
        finally
        {
            SetBusy(false);
            RefreshDashboard();
        }
    }

    private void OnActionsChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (e.NewItems is not null)
            foreach (ActionRow row in e.NewItems)
                row.StatusChanged += (_, _) => RefreshPending();
        RefreshPending();
    }

    private void RefreshPending()
    {
        PendingCount = Actions.Count(a => a.Status == "queued");
        Raise(nameof(ActionsNavLabel));
        Raise(nameof(PendingCard));
    }

    private void RefreshDashboard()
    {
        Raise(nameof(ConnectionCard));
        Raise(nameof(ContractCard));
        Raise(nameof(AssetsCard));
        Raise(nameof(PendingCard));
        Raise(nameof(Step1Text));
        Raise(nameof(Step2Text));
        Raise(nameof(Step3Text));
    }

    private void Record(string @event, string operation, string target, string details) =>
        Audit.Add(new AuditRow
        {
            Time = DateTimeOffset.Now.ToString("HH:mm:ss"),
            Event = @event,
            Operation = operation,
            Target = target,
            Details = details,
        });

    private void SetBusy(bool busy)
    {
        _busy = busy;
        ((RelayCommand)ConnectCommand).Refresh();
        ((RelayCommand)RunCycleCommand).Refresh();
        ((RelayCommand)ApproveCommand).Refresh();
        ((RelayCommand)RefreshLocalCommand).Refresh();
    }
}
