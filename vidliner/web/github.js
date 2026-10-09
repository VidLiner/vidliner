// Public repository metadata only: never use the canvas token or provider credentials.
(function () {
  "use strict";
  const i18n = window.VidLinerI18n;
  const repository = "VidLiner/vidliner";
  const cacheKey = "vidliner.github.stars";
  const ttl = 60 * 60 * 1000;
  const group = document.createElement("div");
  group.className = "github-links";
  const star = document.createElement("a");
  star.id = "githubStar";
  star.href = `https://github.com/${repository}`;
  const history = document.createElement("a");
  history.href = `https://www.star-history.com/#${repository}&Date`;
  for (const link of [star, history]) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    group.appendChild(link);
  }
  document.querySelector(".toolbar").appendChild(group);
  let count = null;

  function render() {
    star.textContent = count === null ? i18n.t("github.star") : i18n.t("github.stars", {
      count: new Intl.NumberFormat(i18n.locale()).format(count),
    });
    history.textContent = i18n.t("github.history");
  }
  function validCount(value) {
    return Number.isSafeInteger(value) && value >= 0;
  }
  window.addEventListener("vidliner:locale-change", render);
  render();

  async function refresh() {
    try {
      const cached = JSON.parse(sessionStorage.getItem(cacheKey) || "null");
      if (cached && cached.repository === repository && validCount(cached.count)
          && Number.isFinite(cached.time) && cached.time <= Date.now()
          && Date.now() - cached.time < ttl) {
        count = cached.count;
        render();
        return;
      }
    } catch (_) { /* storage is optional */ }
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);
    try {
      const response = await fetch(`https://api.github.com/repos/${repository}`, {
        signal: controller.signal,
        credentials: "omit",
        referrerPolicy: "no-referrer",
        headers: { Accept: "application/vnd.github+json" },
      });
      if (!response.ok) return;
      const metadata = await response.json();
      if (!validCount(metadata.stargazers_count)) return;
      count = metadata.stargazers_count;
      render();
      try {
        sessionStorage.setItem(cacheKey, JSON.stringify({ repository, count, time: Date.now() }));
      } catch (_) { /* storage is optional */ }
    } catch (_) { /* keep the repository link usable when offline or rate-limited */ }
    finally { clearTimeout(timeout); }
  }
  refresh();
})();
