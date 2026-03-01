OVERLAY_STYLE = """
    QWidget#overlay {
        background-color: rgba(30, 30, 30, 230);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
"""

OVERLAY_COMPACT_STYLE = """
    QWidget#overlay {
        background-color: rgba(30, 30, 30, 200);
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
"""

MIC_BUTTON_STYLES = {
    "idle": """
        QPushButton {
            background-color: #2d2d2d;
            border: 2px solid #555;
            border-radius: 24px;
            color: #ccc;
            font-size: 18px;
        }
        QPushButton:hover {
            background-color: #3d3d3d;
            border-color: #777;
        }
    """,
    "recording": """
        QPushButton {
            background-color: #e53935;
            border: 2px solid #ff6659;
            border-radius: 24px;
            color: white;
            font-size: 18px;
        }
    """,
    "processing": """
        QPushButton {
            background-color: #ffa726;
            border: 2px solid #ffd95b;
            border-radius: 24px;
            color: white;
            font-size: 18px;
        }
    """,
}

COMBOBOX_STYLE = """
    QComboBox {
        background-color: #3d3d3d;
        color: #ffffff;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 4px 8px;
        min-width: 160px;
        font-size: 12px;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid #aaa;
        margin-right: 6px;
    }
    QComboBox QAbstractItemView {
        background-color: #3d3d3d;
        color: #ffffff;
        selection-background-color: #555;
        border: 1px solid #555;
    }
"""

STATUS_STYLES = {
    "idle": "color: #888888; font-size: 11px;",
    "listening": "color: #4caf50; font-size: 11px; font-weight: bold;",
    "processing": "color: #ffa726; font-size: 11px; font-weight: bold;",
    "done": "color: #2196f3; font-size: 11px;",
    "error": "color: #e53935; font-size: 11px; font-weight: bold;",
}

SETTINGS_BUTTON_STYLE = """
    QPushButton {
        background-color: transparent;
        border: none;
        color: #888;
        font-size: 16px;
        padding: 4px;
    }
    QPushButton:hover {
        color: #ccc;
    }
"""
