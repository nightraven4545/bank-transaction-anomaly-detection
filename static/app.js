// Shared by index.html and dashboard.html. A classic script, so its top-level names are visible to each page's own script.
const $ = (id) => document.getElementById(id);
// Build DOM without innerHTML: uploaded values are untrusted and only ever reach the page as text.
const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...kids.filter((k) => k != null && k !== false));
  return node;
};
const int = new Intl.NumberFormat("en-US");
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`;
const times = (v) => `${v >= 10 ? Math.round(v) : v.toFixed(1)}×`;
const sum = (xs) => xs.reduce((s, x) => s + x, 0);
const median = (xs) => { const s = [...xs].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const money = (v, currency) => currency
  ? new Intl.NumberFormat("en-US", { style: "currency", currency, maximumFractionDigits: Math.abs(v) >= 1000 ? 0 : 2 }).format(v)
  : Number(v).toLocaleString("en-US", { maximumFractionDigits: 2 });
// Dates arrive as "YYYY-MM-DD" or with a time ("2023-04-11 16:29:14"); only the day is shown.
const date = (s) => {
  const d = new Date(`${String(s).slice(0, 10)}T12:00:00`);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
};

const DATASETS = {
  dc: { key: "dc", name: "Washington DC government cards", short: "DC cards", currency: "USD", noun: "payments", dims: ["Agency", "Category"] },
  czech: { key: "czech", name: "Czech bank ledger", short: "Czech bank", currency: "CZK", noun: "withdrawals", dims: ["AccountID", "Narration"] },
  kaggle: { key: "kaggle", name: "Bank transactions demo", short: "Synthetic", currency: "USD", noun: "transactions", dims: ["Channel", "Location"] },
};
const LABEL = {
  TransactionAmount: "Amount", Debit: "Withdrawal",
  amount_vs_agency_median: "× agency's typical purchase", amount_vs_category_median: "× typical for the category",
  category_rarity_in_agency: "Rare category for the agency",
  amount_to_balance: "Amount ÷ balance left", amount_vs_account_median: "× account's usual",
  TransactionDuration: "Session length", LoginAttempts: "Login attempts",
  share_of_balance: "Share of money available", days_since_prev: "Days since last withdrawal",
};

// pandas rank(pct=True) uses the average rank of ties; this matches it so the page agrees with the reason codes.
function bisect(sorted, v, right) {
  let lo = 0, hi = sorted.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < v || (right && sorted[mid] === v)) lo = mid + 1; else hi = mid; }
  return lo;
}

function prepare(body, meta) {
  const rows = body.data.map((r, i) => ({ ...Object.fromEntries(body.columns.map((c, j) => [c, r[j]])), _i: i }));
  const ds = { ...meta, rows, features: body.features, amountKey: body.recipe === "ledger" ? "Debit" : "TransactionAmount" };
  const n = rows.length;
  for (const f of ds.features) {
    const sorted = rows.map((r) => r[f]).sort((a, b) => a - b);
    for (const r of rows) (r._pct ??= {})[f] = ((bisect(sorted, r[f], false) + 1 + bisect(sorted, r[f], true)) / 2 / n) * 100;
  }
  // The amount-only baseline a bank might start with: distance from the median amount, in MADs.
  const amounts = rows.map((r) => r[ds.amountKey]);
  ds.mid = median(amounts);
  const mad = median(amounts.map((a) => Math.abs(a - ds.mid))) || 1;
  rows.forEach((r) => { r._rule = Math.abs(r[ds.amountKey] - ds.mid) / mad; });
  for (const k of ["iforest", "knn", "lof", "_rule"]) {
    [...rows].sort((a, b) => b[k] - a[k]).forEach((r, i) => { (r._rank ??= {})[k] = i + 1; }); // stable: ties keep file order, like the API
  }
  ds.ranked = [...rows].sort((a, b) => a._rank.iforest - b._rank.iforest); // Isolation Forest sets the queue (see report.py)
  // Uploaded dates are untrusted: keep a day only if it is a real calendar date (2023-13-05 or 0000-00-00 would break the time buckets).
  const real = (d) => { const t = Date.parse(`${d}T00:00:00Z`); return /^(19|20)\d\d-\d\d-\d\d$/.test(d) && !Number.isNaN(t) && new Date(t).toISOString().slice(0, 10) === d; };
  for (const r of rows) { const d = String(r.TransactionDate ?? "").slice(0, 10); r._day = real(d) ? d : null; }
  const days = rows.map((r) => r._day).filter(Boolean).sort();
  ds.minDay = days[0] ?? null;
  ds.maxDay = days.at(-1) ?? null;
  return ds;
}

const parseReasons = (s) => (s || "").split(", ").map((x) => x.match(/^(\w+) p(\d+)$/)).filter(Boolean).map(([, f, p]) => ({ f, p: +p }));
const title = (row) => String(row.Vendor ?? row.TransactionID ?? row.Narration ?? `Account ${row.AccountID}`);
const sub = (row) => [row.Agency ?? (row.AccountID != null && row.AccountID !== "statement" ? `Account ${row.AccountID}` : null), row.Category, row.TransactionDate].filter(Boolean).join(" · ");

function sentence(f, row, ds) {
  const v = row[f], amount = row[ds.amountKey], m = (x) => money(x, ds.currency);
  const cat = (row.Category || "").toLowerCase(), above = Math.min(99, Math.floor(row._pct[f]));
  switch (f) {
    case "TransactionAmount": case "Debit": return `${m(v)}, larger than ${above}% of ${ds.noun}`;
    case "amount_vs_agency_median": return `${times(v)} this agency's typical purchase (${m(amount / v)})`;
    case "amount_vs_category_median": return `${times(v)} the typical ${cat} purchase (${m(amount / v)})`;
    case "category_rarity_in_agency": return `Only ${pct(1 - v, 1)} of this agency's purchases are ${cat}`;
    case "amount_to_balance": return `${times(v)} the balance left after paying`;
    case "amount_vs_account_median": return `${times(v)} this account's usual amount (${m(amount / v)})`;
    case "LoginAttempts": return `${v} login attempts before paying (most take 1)`;
    case "TransactionDuration": return `${v}-second session, longer than ${above}% of sessions`;
    case "share_of_balance": return v > 1 ? "More than the money available: the account went overdrawn" : `Took ${pct(v)} of the money available`;
    case "days_since_prev": return `First withdrawal in ${v} days`;
    default: return `${f}: ${v}`;
  }
}

const reasonList = (row, ds) => el("ul", { className: "reasons" }, ...parseReasons(row.reason).map(({ f, p }) =>
  el("li", { className: p < 90 ? "weak" : "", textContent: sentence(f, row, ds) + (p < 90 ? " (weak signal)" : "") })));

const cache = {};
async function load(key) {
  if (!cache[key]) {
    const res = await fetch(`data/${key}.json`);
    if (!res.ok) throw new Error(`Could not load the ${DATASETS[key].name} data (error ${res.status}).`);
    cache[key] = prepare(await res.json(), DATASETS[key]);
  }
  return cache[key];
}

// ---- charts (Apache ECharts) ------------------------------------------------
const MODEL = "Isolation Forest";
const METHOD_NAME = {
  "Isolation Forest": "Isolation Forest (used here)", ensemble: "Average of all three",
  "ensemble: mean of z-scores": "Average of all three (z-scores)", "ensemble: mean of probabilities": "Average of all three (probabilities)",
  "amount only (robust z)": "Amount-only rule", "amount rule": "Amount-only rule", random: "Random checks",
};
const methodName = (m) => METHOD_NAME[m] ?? m;
// [colour token, line width, symbol]. The model is the only coloured line; diamonds are kept for alerts.
const METHOD_STYLE = {
  "Isolation Forest": ["--model", 2.5, "circle"], ensemble: ["--context", 1.5, "triangle"], KNN: ["--context", 1.5, "rect"],
  LOF: ["--context", 1.5, "emptyTriangle"], "amount rule": ["--context", 1.5, "emptyRect"], random: ["--axis", 1.5, "none"],
};

function theme() {
  const axis = {
    axisLine: { lineStyle: { color: token("--axis") } }, axisTick: { show: false },
    axisLabel: { color: token("--muted"), fontFamily: token("--mono"), fontSize: 11, hideOverlap: true },
    splitLine: { lineStyle: { color: token("--hairline") } }, nameTextStyle: { color: token("--ink-2"), fontFamily: token("--sans"), fontSize: 12 },
  };
  return {
    backgroundColor: "transparent", textStyle: { fontFamily: token("--sans"), color: token("--ink-2") },
    categoryAxis: { ...axis, splitLine: { show: false } }, valueAxis: axis, logAxis: axis,
  };
}
const base = (description) => ({
  animation: false,
  aria: { enabled: true, label: { description }, decal: { show: matchMedia("(forced-colors: active), (prefers-contrast: more)").matches } },
});
const tipStyle = () => ({
  confine: true, transitionDuration: 0, backgroundColor: token("--surface"), borderColor: token("--hairline"),
  textStyle: { color: token("--ink"), fontFamily: token("--sans"), fontSize: 13 }, extraCssText: "border-radius:6px;box-shadow:0 6px 20px rgba(0,0,0,.18);",
});
// Tooltip content as DOM nodes: ECharts appends nodes as they are, and only strings go through innerHTML.
const tipBox = (head, ...rows) => el("div", { className: "tt" }, el("strong", { textContent: head }),
  ...rows.map(([k, v]) => el("div", {}, el("span", { textContent: k }), el("b", { textContent: v }))));

const charts = new Map();
const chartResize = new ResizeObserver((entries) => entries.forEach((e) => window.echarts?.getInstanceByDom(e.target)?.resize()));
function chart(id, renderer = "svg", wire) {
  let c = charts.get(id);
  if (!c) {
    c = echarts.init($(id), theme(), { renderer, useDirtyRect: renderer === "canvas" });
    charts.set(id, c);
    chartResize.observe($(id));
    wire?.(c);
  }
  return c;
}
// Colours come from CSS tokens, so a light/dark switch rebuilds every chart from scratch.
function resetCharts() {
  charts.forEach((c, id) => { chartResize.unobserve($(id)); c.dispose(); });
  charts.clear();
}

// Share of the 492 real ULB frauds caught as review capacity grows. Used on the landing page and the Model tab.
function ulbOption(rep, methods, { area = false, endLabel = false } = {}) {
  const u = rep.ulb, i = u.rates.indexOf(0.01), xs = [0, ...u.rates.map((r) => +(r * 100).toFixed(2))];
  return {
    ...base(`Share of ${int.format(u.frauds)} real card frauds caught as review capacity grows. Reviewing 1% of payments: ${methods.filter((m) => m !== "random").map((m) => `${methodName(m)} ${pct(u.recall[m][i])}`).join(", ")}.`),
    grid: { left: 44, right: endLabel ? 112 : 20, top: 44, bottom: 44 },
    legend: { top: 0, left: 0, itemWidth: 16, itemHeight: 8, textStyle: { color: token("--ink-2"), fontSize: 12 } },
    tooltip: { ...tipStyle(), trigger: "axis", formatter: (ps) => {
      const j = ps[0].dataIndex;
      return tipBox(`Reviewing ${xs[j]}% of payments`, ...ps.map((p) => [p.seriesName, `${p.value[1]}% caught`]),
        ...(j ? [[`1 in N model alerts is fraud`, `1 in ${Math.round(1 / u.precision[MODEL][j - 1])}`]] : []));
    } },
    xAxis: { type: "value", min: 0, max: 5, name: "share of payments reviewed", nameLocation: "middle", nameGap: 28, axisLabel: { formatter: "{value}%" } },
    yAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%" } },
    series: methods.map((m) => {
      const [color, width, symbol] = METHOD_STYLE[m], c = token(color), model = m === MODEL;
      return {
        name: methodName(m), type: "line", z: model ? 3 : 2, symbol, symbolSize: model ? 8 : 7,
        data: xs.map((x, j) => [x, j ? +(u.recall[m][j - 1] * 100).toFixed(1) : 0]),
        lineStyle: { color: c, width, type: m === "random" ? "dotted" : "solid" }, itemStyle: { color: c },
        areaStyle: model && area ? { color: c, opacity: 0.1 } : undefined,
        endLabel: model && endLabel ? { show: true, color: token("--ink"), fontWeight: 600, formatter: `${methodName(m).split(" (")[0]}` } : undefined,
        markLine: model ? { silent: true, symbol: "none", label: { show: false }, lineStyle: { color: token("--axis"), type: "dashed" }, data: [{ xAxis: 1 }] } : undefined,
        markPoint: model ? { silent: true, symbol: "circle", symbolSize: 10, itemStyle: { color: c, borderColor: token("--surface"), borderWidth: 2 },
          label: { show: true, position: "right", distance: 8, color: token("--ink"), fontWeight: 600, formatter: `1% reviewed: ${pct(u.recall[m][i])} caught` },
          data: [{ coord: [1, +(u.recall[m][i] * 100).toFixed(1)] }] } : undefined,
      };
    }),
  };
}
