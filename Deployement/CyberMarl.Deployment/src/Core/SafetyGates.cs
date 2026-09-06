// Safety gates mirrored from the Python deployment contract
// (Deployement/validator.py + config.py). The Python sidecar enforces;
// this mirror exists so the UI can label, sort and gate correctly
// WITHOUT reimplementing enforcement. When in doubt, the sidecar wins.

namespace CyberMarl.Deployment.Core;

/// <summary>Execution modes. No fully-automatic destructive enforcement exists.</summary>
public enum DeploymentMode
{
    Shadow,     // log-only, never mutates anything
    Mock,       // simulated effects on policy-view state
    Supervised, // safe ops apply (simulated); destructive queue for approval
    Live        // approved ops execute for real; explicit opt-in only
}

public static class SafetyGates
{
    public static readonly IReadOnlySet<string> DestructiveCommands =
        new HashSet<string> { "Restore", "Remove", "BlockTrafficZone" };

    public static readonly IReadOnlySet<string> InvestigativeCommands =
        new HashSet<string> { "Analyse", "DeployDecoy" };

    public static readonly IReadOnlySet<string> SafeCommands =
        new HashSet<string> { "Sleep", "Monitor", "AllowTrafficZone" };

    public static string RiskOf(string command)
    {
        if (SafeCommands.Contains(command)) return "safe";
        if (InvestigativeCommands.Contains(command)) return "elevated";
        return "destructive";
    }

    public static bool RequiresApproval(string command) =>
        DestructiveCommands.Contains(command) ||
        InvestigativeCommands.Contains(command);

    /// <summary>
    /// Live mode demands an explicit, deliberate opt-in: the operator must
    /// type-confirm, and a real enforcement backend must be configured
    /// (the sidecar refuses live cycles against null backends loudly).
    /// </summary>
    public static bool LiveOptInIsValid(bool typedConfirmation, bool backendConfigured) =>
        typedConfirmation && backendConfigured;
}
