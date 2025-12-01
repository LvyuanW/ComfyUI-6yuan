import json


class PosterTemplateLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template_id": ("STRING", {"default": ""}),
                "templates_json": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("template_layout", "visual_blocks", "font_info")
    FUNCTION = "load"
    CATEGORY = "6yuan/poster"

    def load(self, template_id, templates_json):
        try:
            data = json.loads(templates_json)
        except Exception:
            return (json.dumps({}, ensure_ascii=False), json.dumps([], ensure_ascii=False), json.dumps({}, ensure_ascii=False))

        selected = None
        if isinstance(data, list):
            for t in data:
                if isinstance(t, dict) and str(t.get("id", "")) == str(template_id):
                    selected = t
                    break
        elif isinstance(data, dict) and str(data.get("id", "")) == str(template_id):
            selected = data

        if not isinstance(selected, dict):
            return (json.dumps({}, ensure_ascii=False), json.dumps([], ensure_ascii=False), json.dumps({}, ensure_ascii=False))

        layout = selected.get("layout", {})
        visual_blocks = selected.get("visual_blocks", [])
        style = selected.get("style", {})

        return (
            json.dumps(layout, ensure_ascii=False),
            json.dumps(visual_blocks, ensure_ascii=False),
            json.dumps(style, ensure_ascii=False),
        )


NODE_CLASS_MAPPINGS = {
    "PosterTemplateLoader": PosterTemplateLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PosterTemplateLoader": "Poster Template Loader",
}

