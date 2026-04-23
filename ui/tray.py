"""macOS menu-bar tray icon for gaze corrector controls."""

import threading
import rumps

import config


class GazeTrayApp(rumps.App):
    """System tray application for toggling and adjusting gaze correction."""

    def __init__(self, pipeline):
        super().__init__(
            "Gaze",
            title="👁",
            quit_button=None,  # We'll add our own
        )
        self._pipeline = pipeline
        self._build_menu()

    def _build_menu(self):
        """Build the menu bar items."""
        # Toggle
        toggle_item = rumps.MenuItem("Correction: ON", callback=self._toggle)
        self.menu.add(toggle_item)
        self._toggle_item = toggle_item

        self.menu.add(rumps.separator)

        # Strength submenu
        strength_menu = rumps.MenuItem("Strength")
        self._strength_items = {}
        for label, value in config.STRENGTH_PRESETS.items():
            item = rumps.MenuItem(label, callback=self._set_strength)
            if abs(value - self._pipeline.correction_strength) < 0.01:
                item.state = True
            self._strength_items[label] = item
            strength_menu.add(item)

        self.menu.add(strength_menu)
        self.menu.add(rumps.separator)

        # Quit
        self.menu.add(rumps.MenuItem("Quit", callback=self._quit))

    def _toggle(self, sender):
        self._pipeline.enabled = not self._pipeline.enabled
        sender.title = f"Correction: {'ON' if self._pipeline.enabled else 'OFF'}"

    def _set_strength(self, sender):
        value = config.STRENGTH_PRESETS.get(sender.title, 0.7)
        self._pipeline.correction_strength = value
        # Update checkmarks
        for label, item in self._strength_items.items():
            item.state = (label == sender.title)

    def _quit(self, _):
        self._pipeline.stop()
        rumps.quit_application()


def run_tray(pipeline):
    """Run the tray app (blocks on the main thread)."""
    app = GazeTrayApp(pipeline)
    app.run()


def run_tray_background(pipeline):
    """Run the tray app in a background thread."""
    t = threading.Thread(target=run_tray, args=(pipeline,), daemon=True)
    t.start()
    return t
