#!/usr/bin/env python3
"""
Textual UI for Telegram Analysis Results

A terminal-based interface to browse and visualize analysis results
from the raigem0n pipeline, including summaries, topics, entities,
toxic messages, and style features.

Usage:
    python textual_ui.py
"""

from datetime import datetime
from typing import Dict, List, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button, DataTable, Footer, Header, Label, ListView, 
    ListItem, Placeholder, Static, TabbedContent, TabPane
)
from textual.binding import Binding

from data_loaders import AnalysisRun, discover_runs, load_daily_summary, load_topic_analysis, load_entity_counts, load_toxic_messages, load_style_features
from formatting import (
    fmt_int, fmt_float, fmt_pct, fmt_date, 
    sentiment_color, toxicity_color, confidence_color, count_color,
    format_badge_text, format_run_timestamp, truncate_text
)


class RunListWidget(ListView):
    """Widget displaying list of available analysis runs."""
    
    def __init__(self, runs: List[AnalysisRun], **kwargs):
        super().__init__(**kwargs)
        self.runs = runs
    
    def on_mount(self):
        """Populate runs after the widget is mounted."""
        self.populate_runs()
    
    def populate_runs(self):
        """Populate the ListView with run items."""
        self.clear()
        
        if not self.runs:
            self.append(ListItem(Label("No runs found. Press 'r' to refresh.")))
            return
        
        for run in self.runs:
            # Format run info
            timestamp_str = format_run_timestamp(run.timestamp)
            badge_str = format_badge_text(run.available_files)
            
            # Create run display
            run_label = f"{run.name} [{badge_str}] - {timestamp_str}"
            list_item = ListItem(Label(run_label))
            list_item.run = run  # Attach run data
            self.append(list_item)
    
    def update_runs(self, runs: List[AnalysisRun]):
        """Update the run list with new data."""
        self.runs = runs
        self.populate_runs()
    
    def get_selected_run(self) -> Optional[AnalysisRun]:
        """Get the currently selected analysis run."""
        if self.index is not None and 0 <= self.index < len(self.runs):
            return self.runs[self.index]
        return None


class SummaryPanel(Static):
    """Panel displaying daily summary statistics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📊 Daily Summary", classes="panel-title")
            yield Container(id="summary-kpis")
            yield DataTable(id="summary-table")
    
    def update_data(self, run: AnalysisRun):
        """Update panel with data from selected run."""
        summary_file = run.get_file_path("channel_daily_summary.csv")
        
        if not summary_file:
            self._show_missing_file("channel_daily_summary.csv")
            return
        
        self.data = load_daily_summary(summary_file)
        
        if not self.data:
            self._show_error("Failed to load daily summary data")
            return
        
        self._render_kpis()
        self._render_table()
    
    def _show_missing_file(self, filename: str):
        """Show missing file message."""
        kpis_container = self.query_one("#summary-kpis", Container)
        kpis_container.remove_children()
        kpis_container.mount(Label(f"⚠️  Missing {filename} in this run", classes="missing-file"))
        
        table = self.query_one("#summary-table", DataTable)
        table.clear(columns=True)
    
    def _show_error(self, message: str):
        """Show error message."""
        kpis_container = self.query_one("#summary-kpis", Container)
        kpis_container.mount(Label(f"Error: {message}"))
    
    def _render_kpis(self):
        """Render KPI summary at top of panel."""
        if not self.data:
            return
        
        totals = self.data['totals']
        
        # Create colored KPI display
        sentiment_val = totals['avg_sentiment']
        toxicity_val = totals['avg_toxicity']
        sentiment_color_name = sentiment_color(sentiment_val)
        toxicity_color_name = toxicity_color(toxicity_val)
        
        kpi_text = f"""
📈 Total Messages: [bold cyan]{fmt_int(totals['total_messages'])}[/bold cyan]
😊 Avg Sentiment: [{sentiment_color_name}]{fmt_float(sentiment_val, 3)}[/{sentiment_color_name}]
☠️  Avg Toxicity: [{toxicity_color_name}]{fmt_float(toxicity_val, 3)}[/{toxicity_color_name}]
📅 Date Range: [bright_white]{totals['date_range']}[/bright_white]
🗺️ Days Covered: [bright_cyan]{fmt_int(totals['num_days'])}[/bright_cyan]
        """.strip()
        
        kpis_container = self.query_one("#summary-kpis", Container)
        kpis_container.remove_children()
        kpis_container.mount(Label(kpi_text, markup=True))
    
    def _render_table(self):
        """Render daily data table."""
        if not self.data:
            return
        
        table = self.query_one("#summary-table", DataTable)
        table.clear(columns=True)
        
        # Add columns
        table.add_columns("Date", "Messages", "Sentiment", "Toxicity", "Max Toxic")
        
        # Add rows with color formatting
        for row in self.data['rows']:
            sentiment_val = row['avg_sentiment']
            toxicity_val = row['avg_toxicity']
            max_toxic_val = row['max_toxicity']
            
            # Apply colors based on values
            sentiment_color_name = sentiment_color(sentiment_val)
            toxicity_color_name = toxicity_color(toxicity_val)
            max_toxic_color_name = toxicity_color(max_toxic_val)
            
            table.add_row(
                fmt_date(row['date']),
                fmt_int(row['message_count']),
                f"[{sentiment_color_name}]{fmt_float(sentiment_val, 3)}[/{sentiment_color_name}]",
                f"[{toxicity_color_name}]{fmt_float(toxicity_val, 3)}[/{toxicity_color_name}]",
                f"[{max_toxic_color_name}]{fmt_float(max_toxic_val, 3)}[/{max_toxic_color_name}]"
            )


class TopicsPanel(Static):
    """Panel displaying topic analysis."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("🏷️  Topic Analysis", classes="panel-title")
            yield Container(id="topics-info")
            yield DataTable(id="topics-table")
    
    def update_data(self, run: AnalysisRun):
        """Update panel with data from selected run."""
        topics_file = run.get_file_path("channel_topic_analysis.json")
        
        if not topics_file:
            self._show_missing_file("channel_topic_analysis.json")
            return
        
        self.data = load_topic_analysis(topics_file)
        
        if not self.data:
            self._show_error("Failed to load topic analysis data")
            return
        
        self._render_info()
        self._render_table()
    
    def _show_missing_file(self, filename: str):
        """Show missing file message."""
        info_container = self.query_one("#topics-info", Container)
        info_container.remove_children()
        info_container.mount(Label(f"⚠️  Missing {filename} in this run", classes="missing-file"))
        
        table = self.query_one("#topics-table", DataTable)
        table.clear(columns=True)
    
    def _show_error(self, message: str):
        """Show error message."""
        info_container = self.query_one("#topics-info", Container)
        info_container.mount(Label(f"Error: {message}"))
    
    def _render_info(self):
        """Render topic metadata."""
        if not self.data:
            return
        
        metadata = self.data['metadata']
        confidence_val = metadata['avg_confidence']
        confidence_color_name = confidence_color(confidence_val)
        
        info_text = f"""
🏷️ Topics Found: [bold cyan]{fmt_int(metadata['unique_topics'])}[/bold cyan]
🎯 Avg Confidence: [{confidence_color_name}]{fmt_float(confidence_val, 3)}[/{confidence_color_name}]
⭐ High Confidence: [bright_yellow]{fmt_pct(metadata['confident_percentage'])}[/bright_yellow]
        """.strip()
        
        info_container = self.query_one("#topics-info", Container)
        info_container.remove_children()
        info_container.mount(Label(info_text, markup=True))
    
    def _render_table(self):
        """Render topics table."""
        if not self.data:
            return
        
        table = self.query_one("#topics-table", DataTable)
        table.clear(columns=True)
        
        # Add columns
        table.add_columns("Topic", "Count", "Share", "Confidence")
        
        # Find max count for relative coloring
        max_count = max([t['count'] for t in self.data['topics']], default=1)
        
        # Add rows with colors
        for topic in self.data['topics']:
            count_color_name = count_color(topic['count'], max_count)
            confidence_color_name = confidence_color(topic['confidence'])
            
            table.add_row(
                f"[bold]{topic['topic']}[/bold]",
                f"[{count_color_name}]{fmt_int(topic['count'])}[/{count_color_name}]",
                f"[bright_blue]{fmt_pct(topic['percentage'])}[/bright_blue]",
                f"[{confidence_color_name}]{fmt_float(topic['confidence'], 3)}[/{confidence_color_name}]"
            )


class EntitiesPanel(Static):
    """Panel displaying entity analysis."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.display_limit = 20  # Default to top 20
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("👥 Entity Analysis", classes="panel-title")
            yield Container(id="entities-info")
            yield DataTable(id="entities-table")
    
    def update_data(self, run: AnalysisRun):
        """Update panel with data from selected run."""
        entities_file = run.get_file_path("channel_entity_counts.csv")
        
        if not entities_file:
            self._show_missing_file("channel_entity_counts.csv")
            return
        
        self.data = load_entity_counts(entities_file)
        
        if not self.data:
            self._show_error("Failed to load entity counts data")
            return
        
        self._render_info()
        self._render_table()
    
    def _show_missing_file(self, filename: str):
        """Show missing file message."""
        info_container = self.query_one("#entities-info", Container)
        info_container.remove_children()
        info_container.mount(Label(f"⚠️  Missing {filename} in this run", classes="missing-file"))
        
        table = self.query_one("#entities-table", DataTable)
        table.clear(columns=True)
    
    def _show_error(self, message: str):
        """Show error message."""
        info_container = self.query_one("#entities-info", Container)
        info_container.mount(Label(f"Error: {message}"))
    
    def _render_info(self):
        """Render entity metadata."""
        if not self.data:
            return
        
        metadata = self.data['metadata']
        
        info_text = f"""
👥 Unique Entities: [bold cyan]{fmt_int(metadata['total_entities'])}[/bold cyan]
📊 Total Mentions: [bright_yellow]{fmt_int(metadata['total_mentions'])}[/bright_yellow]
🔍 Showing Top: [bright_green]{fmt_int(min(self.display_limit, len(self.data['entities'])))}[/bright_green]
        """.strip()
        
        info_container = self.query_one("#entities-info", Container)
        info_container.remove_children()
        info_container.mount(Label(info_text, markup=True))
    
    def _render_table(self):
        """Render entities table."""
        if not self.data:
            return
        
        table = self.query_one("#entities-table", DataTable)
        table.clear(columns=True)
        
        # Add columns
        table.add_columns("Entity", "Count", "Share")
        
        # Add rows (limited to display_limit)
        entities_to_show = self.data['entities'][:self.display_limit]
        max_count = max([e['count'] for e in entities_to_show], default=1)
        
        for entity in entities_to_show:
            count_color_name = count_color(entity['count'], max_count)
            
            table.add_row(
                f"[bold]{truncate_text(entity['entity'], 30)}[/bold]",
                f"[{count_color_name}]{fmt_int(entity['count'])}[/{count_color_name}]",
                f"[bright_blue]{fmt_pct(entity['percentage'])}[/bright_blue]"
            )


class ToxicPanel(Static):
    """Panel displaying most toxic messages."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("☠️ Toxic Messages", classes="panel-title")
            yield Container(id="toxic-info")
            yield DataTable(id="toxic-table")
    
    def update_data(self, run: AnalysisRun):
        """Update panel with data from selected run."""
        toxic_file = run.get_file_path("channel_top_toxic_messages.csv")
        
        if not toxic_file:
            self._show_missing_file("channel_top_toxic_messages.csv")
            return
        
        self.data = load_toxic_messages(toxic_file)
        
        if not self.data:
            self._show_error("Failed to load toxic messages data")
            return
        
        self._render_info()
        self._render_table()
    
    def _show_missing_file(self, filename: str):
        """Show missing file message."""
        info_container = self.query_one("#toxic-info", Container)
        info_container.mount(Label(f"Missing {filename} in this run"))
        
        table = self.query_one("#toxic-table", DataTable)
        table.clear()
    
    def _show_error(self, message: str):
        """Show error message."""
        info_container = self.query_one("#toxic-info", Container)
        info_container.mount(Label(f"Error: {message}"))
    
    def _render_info(self):
        """Render toxic messages metadata."""
        if not self.data:
            return
        
        metadata = self.data['metadata']
        
        info_text = f"""
Messages Analyzed: {fmt_int(metadata['total_messages'])}
Max Toxicity: {fmt_float(metadata['max_toxicity'], 3)}
Avg Toxicity: {fmt_float(metadata['avg_toxicity'], 3)}
        """.strip()
        
        info_container = self.query_one("#toxic-info", Container)
        info_container.mount(Label(info_text))
    
    def _render_table(self):
        """Render toxic messages table."""
        if not self.data:
            return
        
        table = self.query_one("#toxic-table", DataTable)
        table.clear(columns=True)
        
        # Add columns
        table.add_columns("Score", "Date", "Author", "Preview", "ID")
        
        # Add rows
        for msg in self.data['messages']:
            table.add_row(
                fmt_float(msg['toxicity_score'], 3),
                fmt_date(msg['date'], short=True),
                truncate_text(msg['author'], 15),
                truncate_text(msg['text_preview'], 50),
                str(msg['msg_id'])
            )


class StylePanel(Static):
    """Panel displaying style features."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("✍️  Style Features", classes="panel-title")
            yield Container(id="style-info")
            yield DataTable(id="style-table")
    
    def update_data(self, run: AnalysisRun):
        """Update panel with data from selected run."""
        style_file = run.get_file_path("channel_style_features.json")
        
        if not style_file:
            self._show_missing_file("channel_style_features.json")
            return
        
        self.data = load_style_features(style_file)
        
        if not self.data:
            self._show_error("Failed to load style features data")
            return
        
        self._render_info()
        self._render_table()
    
    def _show_missing_file(self, filename: str):
        """Show missing file message."""
        info_container = self.query_one("#style-info", Container)
        info_container.mount(Label(f"Missing {filename} in this run"))
        
        table = self.query_one("#style-table", DataTable)
        table.clear()
    
    def _show_error(self, message: str):
        """Show error message."""
        info_container = self.query_one("#style-info", Container)
        info_container.mount(Label(f"Error: {message}"))
    
    def _render_info(self):
        """Render style features metadata."""
        if not self.data:
            return
        
        metadata = self.data['metadata']
        
        info_text = f"""
Features Extracted: {fmt_int(metadata['total_features'])}
Extraction Date: {fmt_date(metadata['extraction_date'])}
        """.strip()
        
        info_container = self.query_one("#style-info", Container)
        info_container.mount(Label(info_text))
    
    def _render_table(self):
        """Render style features table."""
        if not self.data:
            return
        
        table = self.query_one("#style-table", DataTable)
        table.clear(columns=True)
        
        # Add columns
        table.add_columns("Feature", "Value")
        
        # Add rows
        for feature in self.data['features']:
            table.add_row(
                feature['feature'],
                feature['value']
            )


class TelegramAnalysisApp(App):
    """Main Textual application for Telegram analysis results."""
    
    CSS = """
    .panel-title {
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 1;
        margin-bottom: 1;
    }
    
    #run-list {
        width: 30%;
        border: solid $primary;
        margin-right: 1;
    }
    
    #content-area {
        width: 70%;
    }
    
    #summary-kpis {
        background: $surface;
        padding: 1;
        margin: 1;
        border: solid $secondary;
        border-title-align: center;
    }
    
    #topics-info, #entities-info, #toxic-info, #style-info {
        background: $surface;
        padding: 1;
        margin: 1;
        border: solid $secondary;
        border-title-align: center;
    }
    
    .missing-file {
        color: $warning;
        text-style: bold;
        text-align: center;
    }
    
    DataTable {
        height: auto;
        scrollbar-color: $primary;
    }
    
    ListView {
        scrollbar-color: $primary;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_runs", "Refresh"),
        Binding("R", "refresh_run_data", "Reload"),
        Binding("1", "panel_summary", "Summary"),
        Binding("2", "panel_topics", "Topics"),
        Binding("3", "panel_entities", "Entities"),
        Binding("4", "panel_toxic", "Toxic"),
        Binding("5", "panel_style", "Style"),
    ]
    
    def __init__(self):
        super().__init__()
        self.runs = []
        self.selected_run = None
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Horizontal():
            yield Container(
                RunListWidget(self.runs, id="run-list"),
                id="sidebar"
            )
            
            with Container(id="content-area"):
                with TabbedContent():
                    with TabPane("Summary", id="tab-summary"):
                        yield SummaryPanel()
                    with TabPane("Topics", id="tab-topics"):
                        yield TopicsPanel()
                    with TabPane("Entities", id="tab-entities"):
                        yield EntitiesPanel()
                    with TabPane("Toxic", id="tab-toxic"):
                        yield ToxicPanel()
                    with TabPane("Style", id="tab-style"):
                        yield StylePanel()
        
        yield Footer()
    
    def on_mount(self):
        """Initialize app on startup."""
        self.title = "Telegram Analysis Results"
        self.sub_title = "Browse analysis runs"
        self.refresh_runs()
        
        # Set up auto-refresh every 10 seconds
        self.set_interval(10, self.auto_refresh)
    
    def action_refresh_runs(self):
        """Refresh the list of available runs."""
        self.refresh_runs()
        self.notify("Refreshed run list")
    
    def action_refresh_run_data(self):
        """Refresh data for the currently selected run."""
        if self.selected_run:
            # Clear the cache to force reload
            from data_loaders import _load_file_cached
            _load_file_cached.cache_clear()
            
            self.load_run_data(self.selected_run)
            self.notify(f"Refreshed data for {self.selected_run.name}")
        else:
            self.notify("No run selected")
    
    def action_panel_summary(self):
        """Switch to summary panel."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-summary"
    
    def action_panel_topics(self):
        """Switch to topics panel."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-topics"
    
    def action_panel_entities(self):
        """Switch to entities panel."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-entities"
    
    def action_panel_toxic(self):
        """Switch to toxic panel."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-toxic"
    
    def action_panel_style(self):
        """Switch to style panel."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-style"
    
    def auto_refresh(self):
        """Auto-refresh runs in background (non-disruptive)."""
        old_count = len(self.runs)
        new_runs = discover_runs("out")
        
        # Only update if there are new runs
        if len(new_runs) > old_count:
            self.runs = new_runs
            run_list = self.query_one("#run-list", RunListWidget)
            run_list.update_runs(self.runs)
            self.notify(f"Found {len(new_runs) - old_count} new runs")
    
    def refresh_runs(self):
        """Scan for available runs and update the list."""
        self.runs = discover_runs("out")
        
        # Update the run list widget
        run_list = self.query_one("#run-list", RunListWidget)
        run_list.update_runs(self.runs)
        
        # Auto-select first run if available
        if self.runs and not self.selected_run:
            self.selected_run = self.runs[0]
            self.load_run_data(self.selected_run)
    
    def load_run_data(self, run: AnalysisRun):
        """Load data for the specified run into all panels."""
        self.selected_run = run
        self.sub_title = f"Viewing: {run.name}"
        
        # Update all panels
        summary_panel = self.query_one("SummaryPanel", SummaryPanel)
        summary_panel.update_data(run)
        
        topics_panel = self.query_one("TopicsPanel", TopicsPanel)
        topics_panel.update_data(run)
        
        entities_panel = self.query_one("EntitiesPanel", EntitiesPanel)
        entities_panel.update_data(run)
        
        toxic_panel = self.query_one("ToxicPanel", ToxicPanel)
        toxic_panel.update_data(run)
        
        style_panel = self.query_one("StylePanel", StylePanel)
        style_panel.update_data(run)
    
    @on(ListView.Selected)
    def on_run_selected(self, event: ListView.Selected):
        """Handle run selection from the list."""
        run_list = self.query_one("#run-list", RunListWidget)
        selected_run = run_list.get_selected_run()
        
        if selected_run:
            self.load_run_data(selected_run)


def main():
    """Main entry point."""
    app = TelegramAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()