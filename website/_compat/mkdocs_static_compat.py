"""
Compatibility shim: registers the ``static-i18n`` MkDocs plugin entry point and
translates the legacy pre-1.0 config format (dict-based ``languages`` +
``default_language`` key) into the format expected by mkdocs-static-i18n 1.x.

All actual functionality is inherited from ``mkdocs_static_i18n.plugin.I18n``.
"""

from mkdocs_static_i18n.plugin import I18n


class StaticI18n(I18n):
    """Drop-in replacement for the old ``static-i18n`` plugin entry point."""

    def load_config(self, options: dict, config_file_path: str | None = None):
        # Detect old format: languages is a plain dict keyed by locale code.
        if isinstance(options.get("languages"), dict):
            default_lang = options.pop("default_language", "en")
            old_langs = options.pop("languages", {})

            new_langs = []
            for locale, lang_cfg in old_langs.items():
                entry = {
                    "locale": locale,
                    "name": lang_cfg.get("name", locale),
                    "build": lang_cfg.get("build", True),
                    "default": locale == default_lang,
                }
                nav_translations = lang_cfg.get("nav_translations")
                if nav_translations:
                    entry["nav_translations"] = nav_translations
                new_langs.append(entry)

            options["languages"] = new_langs
            options.setdefault("docs_structure", "folder")
            options.setdefault("fallback_to_default", True)
            # Disable material reconfiguration to avoid navigation.instant incompatibility warning.
            options.setdefault("reconfigure_material", False)
            options.setdefault("reconfigure_search", True)

        return super().load_config(options, config_file_path=config_file_path)
