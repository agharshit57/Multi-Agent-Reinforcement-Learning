using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace CyberMarl.Deployment.AvaloniaApp;

/// <summary>Minimal Yes/No modal (Avalonia ships no MessageBox).</summary>
public partial class ConfirmWindow : Window
{
    public bool Result { get; private set; }

    public ConfirmWindow()
    {
        InitializeComponent();
        this.FindControl<Button>("YesButton")!.Click += (_, _) => Close(true);
        this.FindControl<Button>("NoButton")!.Click += (_, _) => Close(false);
    }

    public ConfirmWindow(string title, string message) : this()
    {
        Title = title;
        this.FindControl<TextBlock>("MessageText")!.Text = message;
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}
