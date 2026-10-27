"""Explicit URL normalization and six URL-only lexical features for the CHIIR audit."""
from __future__ import annotations
import ctypes, unicodedata, ipaddress
from urllib.parse import urlsplit, quote

SAFE_URL_CHARS = ":/?#[]@!$&'()*+,;=%-._~"
FEATURES = ["URLLength_recomputed", "HostnameLength", "HostnameLabelCount", "NoOfLettersInURL_recomputed", "NoOfDigitsInURL_recomputed", "NoOfOtherSpecialCharsInURL_recomputed"]

def extract_url_features(raw):
    """Return normalized URL lexical features and an ASCII hostname.

    Order matters: parse the Unicode/NFC URL first, then normalize the host using
    IDNA. Separately percent-encode the full URL for lexical count features.
    Existing percent escapes are preserved and counted literally; they are not
    URL-decoded. Malformed URLs do not raise: hostname-derived counts are zero
    and parse_error records the cause.
    """
    s = unicodedata.normalize("NFC", str(raw).strip())
    normalized_url = quote(s, safe=SAFE_URL_CHARS, encoding="utf-8", errors="strict")
    host = ""; parse_error = ""
    try:
        parts = urlsplit(s)
        host_raw = parts.hostname or ""
        if host_raw:
            host_raw = host_raw.rstrip(".").lower()
            try:
                ipaddress.ip_address(host_raw)
                host = host_raw
            except ValueError:
                try:
                    import idna
                    host = idna.encode(host_raw, uts46=True, std3_rules=True).decode("ascii").lower()
                except Exception:
                    host = host_raw.encode("idna").decode("ascii").lower()
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    if not host and not parse_error:
        parse_error = "No hostname parsed"
    if host:
        hostname_length = len(host)
        hostname_labels = len([label for label in host.split(".") if label])
    else:
        hostname_length = 0; hostname_labels = 0
    letters = sum(ch.isascii() and ch.isalpha() for ch in normalized_url)
    digits = sum(ch.isascii() and ch.isdigit() for ch in normalized_url)
    other = len(normalized_url) - letters - digits
    return {
        "normalised_url": normalized_url,
        "ascii_hostname": host,
        "URLLength_recomputed": len(normalized_url),
        "HostnameLength": hostname_length,
        "HostnameLabelCount": hostname_labels,
        "NoOfLettersInURL_recomputed": letters,
        "NoOfDigitsInURL_recomputed": digits,
        "NoOfOtherSpecialCharsInURL_recomputed": other,
        "parse_error": parse_error,
    }

class PublicSuffixList:
    """Access the system libpsl built-in PSL for registrable-domain grouping."""
    def __init__(self):
        self.lib = ctypes.CDLL("libpsl.so.5")
        self.lib.psl_builtin.restype = ctypes.c_void_p
        self.ctx = self.lib.psl_builtin()
        self.lib.psl_registrable_domain.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.psl_registrable_domain.restype = ctypes.c_char_p
        try:
            self.lib.psl_get_version.restype = ctypes.c_char_p
            self.version = self.lib.psl_get_version().decode("ascii")
        except Exception:
            self.version = "unknown"
    def registrable_domain(self, ascii_hostname):
        if not ascii_hostname:
            return ""
        value = self.lib.psl_registrable_domain(self.ctx, ascii_hostname.encode("ascii", "ignore"))
        return value.decode("ascii") if value else ascii_hostname
