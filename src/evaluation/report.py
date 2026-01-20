class ReportGenerator:
    def __init__(self, metrics):
        self.metrics = metrics

    def generate_report(self):
        report = "Model Evaluation Report\n"
        report += "=" * 30 + "\n"
        for metric, value in self.metrics.items():
            report += f"{metric}: {value:.4f}\n"
        return report

    def save_report(self, filename):
        with open(filename, 'w') as f:
            f.write(self.generate_report())
        print(f"Report saved to {filename}")