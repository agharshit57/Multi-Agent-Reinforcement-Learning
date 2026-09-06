using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using CyberMarl.Deployment.App;

namespace CyberMarl.Deployment.AvaloniaApp;

/// <summary>
/// Thin shell reusing the SHARED MainViewModel (same navigation,
/// checklist, approvals and audit semantics as the WPF console).
/// </summary>
public partial class MainWindow : Window
{
    public MainViewModel ViewModel { get; } = new();

    public MainWindow()
    {
        InitializeComponent();
        ViewModel.ConfirmAction = ConfirmAsync;
        DataContext = ViewModel;
    }

    private async Task<bool> ConfirmAsync(string title, string message)
    {
        var dialog = new ConfirmWindow(title, message);
        return await dialog.ShowDialog<bool>(this);
    }

    public void InitializeBackendFromEnvironment()
    {
        var url = Environment.GetEnvironmentVariable("INFERENCE_URL");
        var token = Environment.GetEnvironmentVariable("INFERENCE_TOKEN");
        ViewModel.InitializeFromEnvironment(url, token);
        if (ViewModel.RefreshLocalCommand.CanExecute(null))
            ViewModel.RefreshLocalCommand.Execute(null);
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}
