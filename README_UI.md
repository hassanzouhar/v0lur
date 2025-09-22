# Telegram Analysis Textual UI 🎯

A beautiful terminal-based interface for browsing and visualizing Telegram analysis results from the raigem0n pipeline.

## Features ✨

- **📊 Interactive Data Visualization**: Browse analysis results in a terminal-based UI
- **🎨 Color-coded Insights**: Visual indicators for sentiment, toxicity, and confidence scores
- **⚡ Real-time Updates**: Auto-refresh detects new analysis runs  
- **🗂️ Multi-panel Views**: Summary, Topics, Entities, Toxic Messages, and Style Features
- **⌨️ Keyboard Navigation**: Fast shortcuts for power users
- **📱 Responsive Layout**: Adaptive interface that works in any terminal size

## Installation 🛠️

The UI is already set up with your project dependencies:

```bash
# Textual is already installed in your .venv
source .venv/bin/activate

# Or use the direct path
.venv/bin/python3.11 textual_ui.py
```

## Quick Start 🚀

1. **Run your analysis** to generate data in `out/` directory
2. **Launch the UI**:
   ```bash
   .venv/bin/python3.11 textual_ui.py
   ```
3. **Navigate** using keyboard shortcuts (see below)

### Preview Demo

```bash
.venv/bin/python3.11 demo_ui.py
```

## Keyboard Shortcuts ⌨️

| Key | Action | Description |
|-----|--------|-------------|
| `q` | Quit | Exit the application |
| `r` | Refresh | Refresh run list (rescan `out/` directory) |
| `R` | Reload | Reload current run data (clears cache) |
| `1` | Summary | Switch to Summary panel |
| `2` | Topics | Switch to Topics panel |
| `3` | Entities | Switch to Entities panel |
| `4` | Toxic | Switch to Toxic Messages panel |
| `5` | Style | Switch to Style Features panel |
| `↑/↓` | Navigate | Select different analysis runs |
| `Enter` | Select | Load selected run |

## Interface Overview 🖥️

### Left Panel: Run List
- **Run badges**: `STEMY` indicators show available data files
  - `S` = Summary data
  - `T` = Topics analysis  
  - `E` = Entity counts
  - `M` = Toxic messages
  - `Y` = Style features
- **Timestamps**: Relative time display (e.g., "4d ago", "Just now")
- **Auto-refresh**: New runs appear automatically

### Right Panel: Tabbed Analysis Views

#### 📊 Summary Panel
- **KPIs**: Total messages, sentiment/toxicity averages, date range
- **Daily Table**: Per-day breakdown with color-coded metrics
- **Colors**: 
  - 🟢 Green = Positive sentiment, low toxicity
  - 🟡 Yellow = Neutral sentiment, moderate toxicity  
  - 🔴 Red = Negative sentiment, high toxicity

#### 🏷️ Topics Panel  
- **Metadata**: Topics found, average confidence, high-confidence percentage
- **Topics Table**: Topic name, count, share percentage, confidence
- **Colors**: Confidence levels from green (high) to red (low)

#### 👥 Entities Panel
- **Statistics**: Unique entities, total mentions
- **Top Entities**: Most frequently mentioned entities with percentages
- **Colors**: Relative frequency coloring

#### ☠️ Toxic Messages Panel *(Coming Soon)*
- **Metadata**: Message count, toxicity statistics
- **Message Table**: Toxicity score, date, author, text preview, message ID
- **Colors**: Toxicity levels from green (safe) to red (toxic)

#### ✍️ Style Features Panel *(Coming Soon)*
- **Features**: Linguistic analysis metrics
- **Key-Value Display**: Readability scores, sentence length, etc.

## Expected File Structure 📁

```
out/
├── run_2024-12-31_235959/     # Analysis run directory
│   ├── channel_daily_summary.csv
│   ├── channel_topic_analysis.json
│   ├── channel_entity_counts.csv
│   ├── channel_top_toxic_messages.csv
│   └── channel_style_features.json
└── another_run/
    └── ...
```

## Color Coding Legend 🎨

### Sentiment Scores
- 🟢 **Green**: ≥ 0.3 (Very Positive)
- 🟢 **Bright Green**: ≥ 0.1 (Positive) 
- 🟡 **Yellow**: -0.1 to 0.1 (Neutral)
- 🟡 **Bright Yellow**: ≥ -0.3 (Slightly Negative)
- 🔴 **Red**: < -0.3 (Very Negative)

### Toxicity Scores  
- 🟢 **Green**: < 0.2 (Safe)
- 🟢 **Bright Green**: 0.2-0.4 (Low)
- 🟡 **Yellow**: 0.4-0.6 (Moderate)
- 🔴 **Red**: 0.6-0.8 (High)
- 🔴 **Bright Red**: ≥ 0.8 (Very High)

### Confidence Scores
- 🟢 **Green**: ≥ 0.8 (Very Confident)
- 🟢 **Bright Green**: 0.6-0.8 (Confident)
- 🟡 **Yellow**: 0.4-0.6 (Moderate)
- 🟡 **Bright Yellow**: 0.2-0.4 (Low)
- 🔴 **Red**: < 0.2 (Very Low)

## Troubleshooting 🔧

### No Runs Found
- **Check**: Ensure `out/` directory exists and contains analysis runs
- **Fix**: Run your analysis pipeline first, or press `r` to refresh

### Missing Data Files
- **Symptom**: Panels show "⚠️ Missing file" warnings
- **Cause**: Incomplete analysis runs (some files weren't generated)
- **Fix**: Re-run analysis or check for errors in the pipeline

### UI Won't Start
- **Check Python version**: Use `.venv/bin/python3.11`
- **Check dependencies**: Textual should be installed in your venv
- **Reinstall**: `pip install "textual>=0.48.0,<1.0.0"`

### Performance Issues
- **Large datasets**: UI includes caching and pagination
- **Memory usage**: Only the selected run is loaded
- **Refresh cache**: Press `R` to clear cache and reload

## Advanced Features 🔥

### Auto-refresh
- Runs are automatically detected every 10 seconds
- New runs appear without restarting the UI
- Current selection is preserved

### Caching System
- File-based caching with modification time checking
- Avoids re-parsing unchanged data files  
- Manual cache clearing with `R` key

### Error Handling
- Graceful handling of missing/malformed files
- User-friendly error messages
- Non-blocking errors (app continues running)

## Development 👨‍💻

### File Structure
```
textual_ui.py       # Main UI application
data_loaders.py     # CSV/JSON parsing utilities  
formatting.py       # Number formatting and colors
demo_ui.py         # Preview demo script
```

### Adding New Panels
1. Create panel class extending `Static`
2. Implement `update_data(run)` method
3. Add to `TabbedContent` in `compose()`
4. Update `load_run_data()` method

### Customizing Colors
Edit the color functions in `formatting.py`:
- `sentiment_color(score)`
- `toxicity_color(score)` 
- `confidence_color(score)`

## License 📄

Same as the parent raigem0n project.

## Support 💬

- **Issues**: Report bugs or feature requests in the project issues
- **Usage**: Reference this README for common questions
- **Development**: Check the code comments for implementation details

---

**Enjoy exploring your Telegram analysis data! 🎉**