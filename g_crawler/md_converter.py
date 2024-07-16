from markdownify import MarkdownConverter
from urllib.parse import unquote


def remove_dot_suffix(text: str) -> str:
    if "." in text:
        return text[: text.rindex(".")]
    return text


def get_sub_path_of_url(url: str) -> str:
    url = url.split("?")[0]
    url = url.split("#")[0]
    return url.split("/")[-1]


class GMarkdownConverter(MarkdownConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert(self, html: str) -> str:
        return super().convert(html)

    def convert_a(self, el, text, convert_as_inline):
        # url decode href
        if "href" in el.attrs:
            href = unquote(el.attrs["href"])
            el.attrs["href"] = href

            if "title" not in el.attrs:
                el.attrs["title"] = get_sub_path_of_url(href)

            del el.attrs["href"]

        return f" {super().convert_a(el, text, convert_as_inline)} "

    def convert_img(self, el, text, convert_as_inline):
        alt = el.attrs.get("alt", None) or ""
        title = el.attrs.get("title", None) or ""
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""

        if not alt and not title:
            return ""

        return "![%s](%s)" % (alt, title_part)
