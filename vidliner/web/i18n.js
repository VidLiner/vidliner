(function (global) {
  "use strict";

  const localeElement = document.getElementById("locales");
  const resources = localeElement ? JSON.parse(localeElement.textContent || "{}") : {};
  const supported = Object.keys(resources);
  let locale = null;

  function normalize(value) {
    if (!value) return "en-US";
    const exact = supported.find((candidate) => candidate.toLowerCase() === value.toLowerCase());
    if (exact) return exact;
    const language = value.split("-")[0].toLowerCase();
    return supported.find((candidate) => candidate.split("-")[0].toLowerCase() === language) || "en-US";
  }

  function detect() {
    try {
      const stored = global.localStorage.getItem("vidliner.locale");
      if (stored) return normalize(stored);
    } catch (_) { /* storage is optional in embedded previews */ }
    const languages = global.navigator.languages || [global.navigator.language];
    return normalize(languages.find(Boolean) || "en-US");
  }

  function format(template, values) {
    return String(template).replace(/\{([\w]+)\}/g, (_, key) => {
      return values && values[key] !== undefined ? String(values[key]) : `{${key}}`;
    });
  }

  function t(key, values) {
    const active = resources[locale] || resources["en-US"] || {};
    const fallback = resources["en-US"] || {};
    const template = active[key] ?? fallback[key] ?? values?.default ?? key;
    return format(template, values);
  }

  function apply(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-i18n]").forEach((element) => {
      element.textContent = t(element.dataset.i18n);
    });
    scope.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
      element.setAttribute("placeholder", t(element.dataset.i18nPlaceholder));
    });
    scope.querySelectorAll("[data-i18n-label]").forEach((element) => {
      element.setAttribute("aria-label", t(element.dataset.i18nLabel));
    });
    const selector = document.getElementById("localeSelect");
    if (selector) selector.value = locale;
    document.documentElement.lang = locale;
    document.documentElement.dir = /^(ar|fa|he)(-|$)/i.test(locale) ? "rtl" : "ltr";
  }

  function setLocale(value) {
    locale = normalize(value);
    try { global.localStorage.setItem("vidliner.locale", locale); } catch (_) { /* optional */ }
    apply();
    global.dispatchEvent(new CustomEvent("vidliner:locale-change", { detail: { locale } }));
  }

  function init() {
    if (!locale) locale = detect();
    const selector = document.getElementById("localeSelect");
    if (selector && !selector.dataset.bound) {
      selector.dataset.bound = "1";
      selector.addEventListener("change", () => setLocale(selector.value));
    }
    apply();
    return locale;
  }

  global.VidLinerI18n = { apply, format, init, locale: () => locale, setLocale, t };
})(window);
