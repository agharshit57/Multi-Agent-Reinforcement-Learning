using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;

namespace CyberMarl.Deployment.AvaloniaApp;

public partial class App : Application
{
    private static Mutex? _instanceMutex;

    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        // Single-instance guard (plain name: portable to Linux; a Mutex
        // failure degrades to "no guard", never to "no app").
        try
        {
            _instanceMutex = new Mutex(
                initiallyOwned: true,
                name: "CyberMarlDeploymentConsole",
                createdNew: out bool createdNew);
            if (!createdNew)
            {
                if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
                    desktop.Shutdown();
                return;
            }
        }
        catch { _instanceMutex = null; }

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktopLifetime)
        {
            var window = new MainWindow();
            window.InitializeBackendFromEnvironment();
            desktopLifetime.MainWindow = window;
        }
        base.OnFrameworkInitializationCompleted();
    }
}
