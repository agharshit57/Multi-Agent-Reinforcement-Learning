using System.Windows;

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
}
