using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;

namespace CyberMarl.Deployment.App;

/// <summary>
/// Thin shell: owns the MainViewModel, resolves env config, and hosts the
/// live-mode confirmation dialog. No policy, topology or enforcement logic
/// lives here (see ViewModels.cs + Core).
/// </summary>
public partial class MainWindow : Window
{
    public MainViewModel ViewModel { get; } = new();

    public MainWindow()
    {
        InitializeComponent();
        ViewModel.ConfirmAction = (title, message) => Task.FromResult(
            MessageBox.Show(this, message, title,
                MessageBoxButton.YesNo, MessageBoxImage.Warning) == MessageBoxResult.Yes);
        DataContext = ViewModel;
    }

    public void InitializeBackendFromEnvironment()
    {
        var url = Environment.GetEnvironmentVariable("INFERENCE_URL");
        var token = Environment.GetEnvironmentVariable("INFERENCE_TOKEN");
        ViewModel.InitializeFromEnvironment(url, token);
        if (ViewModel.RefreshLocalCommand.CanExecute(null))
            ViewModel.RefreshLocalCommand.Execute(null);
    }

    protected override void OnSourceInitialized(EventArgs e)
    {
        base.OnSourceInitialized(e);
        EnableImmersiveDarkTitleBar();
    }

    /// <summary>
    /// Black native title bar (DWM immersive dark mode) so the OS chrome
    /// matches the app theme. Best-effort: any failure keeps the default
    /// title bar, never breaks startup.
    /// </summary>
    private void EnableImmersiveDarkTitleBar()
    {
        try
        {
            var hwnd = new WindowInteropHelper(this).Handle;
            if (hwnd == IntPtr.Zero) return;
            int enabled = 1;
            // DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (19 on older Win10).
            if (DwmSetWindowAttribute(hwnd, 20, ref enabled,
                                      Marshal.SizeOf<int>()) != 0)
                DwmSetWindowAttribute(hwnd, 19, ref enabled,
                                      Marshal.SizeOf<int>());
        }
        catch { /* default title bar */
        }
    }

    [DllImport("dwmapi.dll", PreserveSig = true)]
    private static extern int DwmSetWindowAttribute(
        IntPtr hwnd, int attr, ref int value, int size);
}
