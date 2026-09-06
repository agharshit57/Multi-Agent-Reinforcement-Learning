using System.Windows;

namespace CyberMarl.Deployment.App;

/// <summary>
/// Single-instance startup. There is deliberately NO StartupUri in
/// App.xaml: the window is created exactly once here. (A StartupUri plus
/// this manual creation is what used to open the console twice.)
/// </summary>
public partial class App : Application
{
    private static System.Threading.Mutex? _instanceMutex;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        _instanceMutex = new System.Threading.Mutex(
            initiallyOwned: true,
            name: @"Global\CyberMarlDeploymentConsole",
            createdNew: out bool createdNew);
        if (!createdNew)
        {
            MessageBox.Show(
                "The Cyber MARL console is already running.",
                "Cyber MARL", MessageBoxButton.OK, MessageBoxImage.Information);
            Shutdown();
            return;
        }
        var window = new MainWindow();
        window.InitializeBackendFromEnvironment();
        MainWindow = window;
        window.Show();
    }
}
