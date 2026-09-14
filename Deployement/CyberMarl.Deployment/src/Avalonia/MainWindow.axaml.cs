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
        // Logo URIs probed in order: the linked resource name depends on
        // build-time Link handling, so never assume exactly one shape.
        // A missing logo hides the image; the app never depends on it.
        var logoUri = FindLogoUri();
        var logoImage = this.FindControl<Image>("LogoImage");
        if (logoImage is not null)
        {
            try
            {
                if (logoUri is not null)
                    logoImage.Source = new Avalonia.Media.Imaging.Bitmap(
                        Avalonia.Platform.AssetLoader.Open(logoUri));
                else
                    logoImage.IsVisible = false;
            }
            catch { logoImage.IsVisible = false; }
        }
        try
        {
            if (logoUri is not null)
                Icon = new WindowIcon(
                    Avalonia.Platform.AssetLoader.Open(logoUri));
        }
        catch { /* window/taskbar icon is decoration, never fatal */ }
        ViewModel.ConfirmAction = ConfirmAsync;
        DataContext = ViewModel;
    }

    private static Uri? FindLogoUri()
    {
        foreach (var path in new[]
                 {
                     "avares://CyberMarl.Deployment.AvaloniaApp/Assets/logo.png",
                     "avares://CyberMarl.Deployment.AvaloniaApp/logo.png",
                 })
        {
            try
            {
                var uri = new Uri(path);
                if (Avalonia.Platform.AssetLoader.Exists(uri))
                    return uri;
            }
            catch { /* try the next shape */ }
        }
        return null;
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
