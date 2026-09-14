// Integration boundary: DTOs + HTTP client for the Python sidecar
// (Deployement/inference_service.py, schema "real-first/v1").
//
// The trained PyTorch MAPPO stack stays in Python. This client sends
// REAL topology/telemetry and receives policy views, decisions and
// real-action plans. Timeouts are enforced; secrets are Bearer tokens
// (never logged); every error surfaces as InferenceException.

using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace CyberMarl.Deployment.Core;

public sealed class InferenceException : Exception
{
    public InferenceException(string message) : base(message) { }
    public InferenceException(string message, Exception inner) : base(message, inner) { }
}

// ------------------------------------------------------------ DTOs --

public sealed record SecurityEventDto(
    [property: JsonPropertyName("kind")] string Kind,
    [property: JsonPropertyName("severity")] string Severity,
    [property: JsonPropertyName("details")] string Details,
    [property: JsonPropertyName("timestamp")] double Timestamp);

public sealed record HostTelemetryDto(
    [property: JsonPropertyName("processes")] List<string> Processes,
    [property: JsonPropertyName("connections")] List<string> Connections,
    [property: JsonPropertyName("sessions")] List<string> Sessions,
    [property: JsonPropertyName("up")] bool Up,
    [property: JsonPropertyName("events")] List<SecurityEventDto> Events,
    [property: JsonPropertyName("services")] List<string>? Services = null);

public sealed record RealBatchDto(
    [property: JsonPropertyName("timestamp")] double Timestamp,
    [property: JsonPropertyName("source")] string Source,
    [property: JsonPropertyName("hosts")] Dictionary<string, HostTelemetryDto> Hosts,
    [property: JsonPropertyName("notes")] string Notes = "",
    [property: JsonPropertyName("partial_errors")] List<Dictionary<string, string>>? PartialErrors = null);

public sealed record RealAssetDto(
    [property: JsonPropertyName("asset_id")] string AssetId,
    [property: JsonPropertyName("hostname")] string Hostname,
    [property: JsonPropertyName("ips")] List<string> Ips,
    [property: JsonPropertyName("macs")] List<string> Macs,
    [property: JsonPropertyName("os_name")] string OsName,
    [property: JsonPropertyName("os_version")] string OsVersion,
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("segment_id")] string SegmentId,
    [property: JsonPropertyName("coverage")] string Coverage = "unknown",
    [property: JsonPropertyName("coverage_detail")] string CoverageDetail = "",
    [property: JsonPropertyName("services")] List<string>? Services = null,
    [property: JsonPropertyName("observed_this_cycle")] bool ObservedThisCycle = false);

public sealed record RealSegmentDto(
    [property: JsonPropertyName("segment_id")] string SegmentId,
    [property: JsonPropertyName("name")] string Name = "",
    [property: JsonPropertyName("kind")] string Kind = "unknown",
    [property: JsonPropertyName("cidrs")] List<string>? Cidrs = null);

public sealed record RealLinkDto(
    [property: JsonPropertyName("asset_id")] string AssetId,
    [property: JsonPropertyName("peer")] string Peer,
    [property: JsonPropertyName("via")] string Via = "unknown");

public sealed record RealTopologyDto(
    [property: JsonPropertyName("assets")] List<RealAssetDto> Assets,
    [property: JsonPropertyName("segments")] List<RealSegmentDto> Segments,
    [property: JsonPropertyName("source")] string Source = "csharp",
    [property: JsonPropertyName("discovered_at")] double DiscoveredAt = 0,
    [property: JsonPropertyName("links")] List<RealLinkDto>? Links = null);

public sealed record PolicyBindingDto(
    [property: JsonPropertyName("asset_id")] string AssetId,
    [property: JsonPropertyName("agent")] string Agent,
    [property: JsonPropertyName("cc4")] string PolicySlot);

public sealed record PolicyMappingDto(
    [property: JsonPropertyName("bindings")] List<PolicyBindingDto> Bindings,
    [property: JsonPropertyName("unpopulated")] List<string> Unpopulated,
    [property: JsonPropertyName("overflow")] List<string> Overflow,
    [property: JsonPropertyName("digest")] string Digest);

public sealed record DecisionDto(
    [property: JsonPropertyName("agent_id")] int AgentId,
    [property: JsonPropertyName("action")] int Action,
    [property: JsonPropertyName("label")] string Label,
    [property: JsonPropertyName("operation")] string? Operation = null,
    [property: JsonPropertyName("status")] string? Status = null,
    [property: JsonPropertyName("command")] string? Command = null);

public sealed record RealActionDto(
    [property: JsonPropertyName("asset_id")] string AssetId,
    [property: JsonPropertyName("hostname")] string Hostname,
    [property: JsonPropertyName("operation")] string Operation,
    [property: JsonPropertyName("risk")] string Risk,
    [property: JsonPropertyName("requires_approval")] bool RequiresApproval,
    [property: JsonPropertyName("blocked")] bool Blocked,
    [property: JsonPropertyName("block_reason")] string BlockReason,
    [property: JsonPropertyName("description")] string Description);

public sealed record ContractInfo(
    [property: JsonPropertyName("schema")] string Schema,
    [property: JsonPropertyName("obs_dim")] int ObsDim,
    [property: JsonPropertyName("action_dim")] int ActionDim,
    [property: JsonPropertyName("num_host_targets")] int NumHostTargets,
    [property: JsonPropertyName("num_agents")] int NumAgents);

public sealed record HealthInfo(
    [property: JsonPropertyName("ok")] bool Ok,
    [property: JsonPropertyName("engine")] string Engine,
    [property: JsonPropertyName("mode")] string? Mode,
    [property: JsonPropertyName("mapping")] string? Mapping);

public sealed record CycleResult(
    PolicyMappingDto Mapping,
    List<DecisionDto> Records,
    List<RealActionDto> RealActions,
    RealTopologyDto Topology);

public sealed record ApprovalOutcome(
    string Operation,
    bool Applied,
    string Error,
    string Details);

/// <summary>One server-side queued approval (stable id, not a position).</summary>
public sealed record ApprovalDto(
    [property: JsonPropertyName("position")] int Position,
    [property: JsonPropertyName("approval_id")] string? ApprovalId,
    [property: JsonPropertyName("operation")] string Operation,
    [property: JsonPropertyName("command")] string Command,
    [property: JsonPropertyName("target")] string Target,
    [property: JsonPropertyName("asset")] string Asset,
    [property: JsonPropertyName("risk")] string Risk,
    [property: JsonPropertyName("attempts")] int Attempts);

// ---------------------------------------------------------- client --

/// <summary>HTTP client for the Python inference sidecar (loopback).</summary>
public sealed class InferenceClient : IDisposable
{
    public const string ExpectedSchema = "real-first/v1";

    private readonly HttpClient _http;
    private bool _disposed;

    public InferenceClient(string baseUrl, string token, TimeSpan? timeout = null)
    {
        if (string.IsNullOrWhiteSpace(token))
            throw new InferenceException("sidecar token is required (INFERENCE_TOKEN)");
        _http = new HttpClient
        {
            BaseAddress = new Uri(baseUrl.TrimEnd('/')),
            Timeout = timeout ?? TimeSpan.FromSeconds(30),
        };
        _http.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", token);
    }

    public async Task<ContractInfo> GetContractAsync(CancellationToken ct = default)
    {
        var contract = await GetAsync<ContractInfo>("/contract", ct);
        if (contract.Schema != ExpectedSchema)
            throw new InferenceException(
                $"sidecar schema {contract.Schema} != {ExpectedSchema}; refusing to serve mismatched contract");
        if (contract.ObsDim != 210 || contract.ActionDim != 242 || contract.NumHostTargets != 137)
            throw new InferenceException(
                $"sidecar contract obs={contract.ObsDim}/act={contract.ActionDim}/vocab={contract.NumHostTargets} " +
                "does not match the frozen 210/242/137 checkpoint contract");
        return contract;
    }

    public Task<HealthInfo> GetHealthAsync(CancellationToken ct = default) =>
        GetAsync<HealthInfo>("/health", ct);

    public async Task<CycleResult> RunCycleAsync(
        RealTopologyDto topology, RealBatchDto batch, CancellationToken ct = default)
    {
        // StringContent (not PostAsJsonAsync): JsonContent streams with
        // chunked transfer encoding (no Content-Length), which stock
        // HTTP servers such as Python's http.server cannot parse --
        // the sidecar answers those with 400. Byte-backed content
        // always carries an explicit Content-Length.
        var json = System.Text.Json.JsonSerializer.Serialize(
            new { topology, batch });
        using var content = new StringContent(
            json, System.Text.Encoding.UTF8, "application/json");
        using var response = await _http.PostAsync("/cycle", content, ct);
        var doc = await ReadAsync<JsonCycleResponse>(response);
        return new CycleResult(doc.Mapping, doc.Records, doc.RealActions,
            doc.RealTopology ?? new RealTopologyDto(
                new List<RealAssetDto>(), new List<RealSegmentDto>()));
    }

    /// <summary>
    /// Approve one queued decision by STABLE id (never by position:
    /// the server queue persists across cycles while list positions
    /// shift, so position-based approval can pop the wrong decision).
    /// </summary>
    public async Task<ApprovalOutcome> ApproveAsync(
        string approvalId, string approver, CancellationToken ct = default)
    {
        if (string.IsNullOrEmpty(approvalId))
            throw new InferenceException(
                "refusing approval without a stable approval_id " +
                "(position-based approval can pop the wrong decision)");
        // StringContent for the same Content-Length reason as /cycle.
        var json = System.Text.Json.JsonSerializer.Serialize(
            new { approval_id = approvalId, approver });
        using var content = new StringContent(
            json, System.Text.Encoding.UTF8, "application/json");
        using var response = await _http.PostAsync("/approve", content, ct);
        var doc = await ReadAsync<JsonApproveResponse>(response);
        return new ApprovalOutcome(doc.Operation, doc.Applied, doc.Error, doc.Details);
    }

    /// <summary>
    /// Full pending-approvals snapshot (stable ids). The UI renders
    /// THIS list -- never a per-cycle positional list.
    /// </summary>
    public async Task<List<ApprovalDto>> GetApprovalsAsync(
        CancellationToken ct = default)
    {
        var doc = await GetAsync<JsonApprovalsResponse>("/approvals", ct);
        return doc.Approvals ?? new List<ApprovalDto>();
    }

    private async Task<T> GetAsync<T>(string path, CancellationToken ct)
    {
        using var response = await _http.GetAsync(path, ct);
        return await ReadAsync<T>(response);
    }

    private static async Task<T> ReadAsync<T>(HttpResponseMessage response)
    {
        string body;
        try { body = await response.Content.ReadAsStringAsync(); }
        catch (Exception ex) { throw new InferenceException($"sidecar unreadable: {ex.Message}", ex); }
        if (!response.IsSuccessStatusCode)
            throw new InferenceException($"sidecar {(int)response.StatusCode}: {ExtractError(body)}");
        try
        {
            return System.Text.Json.JsonSerializer.Deserialize<T>(
                body, new System.Text.Json.JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                }) ?? throw new InferenceException("sidecar returned empty body");
        }
        catch (InferenceException) { throw; }
        catch (Exception ex) { throw new InferenceException($"sidecar bad schema: {ex.Message}", ex); }
    }

    private static string ExtractError(string body)
    {
        try
        {
            using var doc = System.Text.Json.JsonDocument.Parse(body);
            if (doc.RootElement.TryGetProperty("error", out var error))
                return error.GetString() ?? body;
        }
        catch { /* fall through */ }
        return body;
    }

    public void Dispose()
    {
        if (!_disposed) { _http.Dispose(); _disposed = true; }
    }

    private sealed record JsonCycleResponse(
        [property: JsonPropertyName("mapping")] PolicyMappingDto Mapping,
        [property: JsonPropertyName("records")] List<DecisionDto> Records,
        [property: JsonPropertyName("real_actions")] List<RealActionDto> RealActions,
        [property: JsonPropertyName("real_topology")] RealTopologyDto? RealTopology);

    private sealed record JsonApproveResponse(
        [property: JsonPropertyName("operation")] string Operation,
        [property: JsonPropertyName("applied")] bool Applied,
        [property: JsonPropertyName("error")] string Error,
        [property: JsonPropertyName("details")] string Details);

    private sealed record JsonApprovalsResponse(
        [property: JsonPropertyName("approvals")] List<ApprovalDto>? Approvals);

    // ------------------------------------------------- DTO mapping --
    public static RealTopology ToModel(RealTopologyDto dto) =>
        new(
            dto.Assets.Select(a => new RealAsset(
                a.AssetId, a.Hostname, a.Ips, a.Macs,
                a.OsName, a.OsVersion, a.Role, a.SegmentId,
                a.Coverage, a.CoverageDetail, a.Services)).ToList(),
            dto.Segments.Select(s => new RealSegment(
                s.SegmentId, s.Name, s.Kind, s.Cidrs ?? new List<string>())).ToList(),
            dto.Source,
            DateTimeOffset.FromUnixTimeSeconds((long)dto.DiscoveredAt),
            (dto.Links ?? new List<RealLinkDto>()).Select(l => new RealLink(
                l.AssetId, l.Peer, l.Via)).ToList());

    public static PolicyMapping ToModel(PolicyMappingDto dto) =>
        new(
            dto.Bindings.Select(b => new PolicyBinding(
                b.AssetId, "", new List<string>(), b.Agent, b.PolicySlot, "")).ToList(),
            dto.Unpopulated,
            dto.Overflow,
            dto.Digest);
}
