const groups = [
  {
    title: "Schrödinger Bridge",
    tracks: [["sb-1", "1 step"], ["sb-5", "5 steps"], ["sb-50", "50 steps"]],
    proposed: true,
  },
  {
    title: "Gaussian Bridge",
    tracks: [["gb-1", "1 step"], ["gb-5", "5 steps"], ["gb-50", "50 steps"]],
    proposed: true,
  },
  {
    title: "Gaussian Tube",
    tracks: [["tube-1", "1 step"], ["tube-5", "5 steps"], ["tube-50", "50 steps"]],
    proposed: true,
  },
  {
    title: "Training-free baselines",
    tracks: [["knn", "kNN"], ["kdot", "kDOT"], ["mkl", "MKL"]],
  },
  {
    title: "Learned baselines",
    tracks: [["freevc", "FreeVC"], ["ph-expanded", "Phoneme Hallucinator"]],
  },
];

const resultRows = [
  { label: "SB · 1", group: "proposed", wer: [0.23, "0.23 [0.22, 0.24]"], utmos: [3.04, "3.04 ± 0.42"], fadSource: [0.85, "0.85"], fadTarget: [1.03, "1.03"], deltaFad: [0.18, "0.18"], sourceSim: [0.55, "0.55 [0.54, 0.56]"], targetSim: [0.30, "0.30 [0.29, 0.31]"] },
  { label: "SB · 5", group: "proposed", wer: [0.23, "0.23 [0.22, 0.24]"], utmos: [3.11, "3.11 ± 0.43"], fadSource: [0.78, "0.78"], fadTarget: [0.96, "0.96"], deltaFad: [0.18, "0.18"], sourceSim: [0.64, "0.64 [0.63, 0.65]"], targetSim: [0.28, "0.28 [0.27, 0.28]"] },
  { label: "SB · 50", group: "proposed", wer: [0.23, "0.23 [0.22, 0.24]"], utmos: [3.11, "3.11 ± 0.42"], fadSource: [0.75, "0.75"], fadTarget: [0.93, "0.93"], deltaFad: [0.18, "0.18"], sourceSim: [0.65, "0.65 [0.64, 0.66]"], targetSim: [0.27, "0.27 [0.26, 0.27]"] },
  { label: "GB · 1", group: "proposed", wer: [0.23, "0.23 [0.23, 0.24]"], utmos: [3.12, "3.12 ± 0.37"], fadSource: [0.93, "0.93"], fadTarget: [1.07, "1.07"], deltaFad: [0.14, "0.14"], sourceSim: [0.55, "0.55 [0.54, 0.56]"], targetSim: [0.29, "0.29 [0.28, 0.30]"] },
  { label: "GB · 5", group: "proposed", wer: [0.23, "0.23 [0.22, 0.24]"], utmos: [3.11, "3.11 ± 0.42"], fadSource: [0.81, "0.81"], fadTarget: [0.98, "0.98"], deltaFad: [0.17, "0.17"], sourceSim: [0.64, "0.64 [0.63, 0.65]"], targetSim: [0.27, "0.27 [0.26, 0.28]"] },
  { label: "GB · 50", group: "proposed", wer: [0.23, "0.23 [0.22, 0.24]"], utmos: [3.10, "3.10 ± 0.43"], fadSource: [0.78, "0.78"], fadTarget: [0.95, "0.95"], deltaFad: [0.17, "0.17"], sourceSim: [0.65, "0.65 [0.65, 0.66]"], targetSim: [0.26, "0.26 [0.25, 0.27]"] },
  { label: "Tube · 1", group: "proposed", wer: [0.23, "0.23 [0.22, 0.24]"], utmos: [3.08, "3.08 ± 0.38"], fadSource: [0.91, "0.91"], fadTarget: [1.04, "1.04"], deltaFad: [0.13, "0.13"], sourceSim: [0.55, "0.55 [0.54, 0.56]"], targetSim: [0.29, "0.29 [0.29, 0.30]"] },
  { label: "Tube · 5", group: "proposed", wer: [0.23, "0.23 [0.22, 0.24]"], utmos: [3.09, "3.09 ± 0.40"], fadSource: [0.80, "0.80"], fadTarget: [0.96, "0.96"], deltaFad: [0.16, "0.16"], sourceSim: [0.64, "0.64 [0.63, 0.65]"], targetSim: [0.27, "0.27 [0.26, 0.27]"] },
  { label: "Tube · 50", group: "proposed", wer: [0.23, "0.23 [0.22, 0.24]"], utmos: [3.12, "3.12 ± 0.42"], fadSource: [0.75, "0.75"], fadTarget: [0.92, "0.92"], deltaFad: [0.17, "0.17"], sourceSim: [0.66, "0.66 [0.65, 0.66]"], targetSim: [0.26, "0.26 [0.25, 0.26]"] },
  { label: "kNN", group: "discrete", wer: [0.50, "0.50 [0.47, 0.53]"], utmos: [2.50, "2.50 ± 0.51"], fadSource: [1.22, "1.22"], fadTarget: [1.41, "1.41"], deltaFad: [0.19, "0.19"], sourceSim: [0.18, "0.18 [0.17, 0.18]"], targetSim: [0.48, "0.48 [0.47, 0.49]"] },
  { label: "kDOT", group: "discrete", wer: [0.48, "0.48 [0.46, 0.50]"], utmos: [2.51, "2.51 ± 0.47"], fadSource: [1.12, "1.12"], fadTarget: [1.29, "1.29"], deltaFad: [0.17, "0.17"], sourceSim: [0.17, "0.17 [0.16, 0.17]"], targetSim: [0.51, "0.51 [0.50, 0.51]"] },
  { label: "MKL", group: "discrete", wer: [0.72, "0.72 [0.61, 0.82]"], utmos: [2.36, "2.36 ± 0.64"], fadSource: [3.51, "3.51"], fadTarget: [3.70, "3.70"], deltaFad: [0.19, "0.19"], sourceSim: [0.28, "0.28 [0.27, 0.30]"], targetSim: [0.27, "0.27 [0.26, 0.28]"] },
  { label: "FreeVC", group: "learned", wer: [0.24, "0.24 [0.23, 0.26]"], utmos: [2.86, "2.86 ± 0.39"], fadSource: [2.58, "2.58"], fadTarget: [2.80, "2.80"], deltaFad: [0.22, "0.22"], sourceSim: [0.31, "0.31 [0.30, 0.31]"], targetSim: [0.28, "0.28 [0.27, 0.29]"] },
  { label: "Phoneme Hall.", group: "learned", wer: [0.25, "0.25 [0.24, 0.26]"], utmos: [2.86, "2.86 ± 0.43"], fadSource: [1.65, "1.65"], fadTarget: [1.79, "1.79"], deltaFad: [0.14, "0.14"], sourceSim: [0.34, "0.34 [0.33, 0.34]"], targetSim: [0.36, "0.36 [0.35, 0.37]"] },
];

const metricMeta = {
  wer: { title: "Word error rate", direction: "Lower is better" },
  utmos: { title: "UTMOSv2", direction: "Higher is better" },
  fadSource: { title: "FAD to source distribution", direction: "" },
  fadTarget: { title: "FAD to target distribution", direction: "" },
  deltaFad: { title: "Target-relative FAD", direction: "Lower is better" },
  sourceSim: { title: "Source-speaker similarity", direction: "Lower is better" },
  targetSim: { title: "Target-speaker similarity", direction: "Higher is better" },
};

let examples = [];
let selectedExample = 0;
let selectedMetric = "wer";

function audioCard(example, id, label, meta, highlight = false) {
  const path = `audio/${example.id}/${example.files[id]}`;
  return `
    <article class="audio-card${highlight ? " highlight" : ""}">
      <div class="audio-card-head"><strong>${label}</strong><small>${meta}</small></div>
      <audio controls preload="none" src="${path}">Your browser does not support audio playback.</audio>
    </article>`;
}

function stopAudio() {
  document.querySelectorAll("audio").forEach((audio) => {
    audio.pause();
    audio.currentTime = 0;
  });
}

function renderExample() {
  const example = examples[selectedExample];
  if (!example) return;
  stopAudio();
  document.querySelectorAll("[data-example]").forEach((tab, index) => {
    tab.setAttribute("aria-selected", String(index === selectedExample));
  });
  document.querySelector("#pair-label").textContent =
    example.pair_label || `${example.source_id} → ${example.target_id}`;
  document.querySelector("#transcript").textContent = `“${example.transcript}”`;
  document.querySelector("#reference-row").innerHTML =
    audioCard(example, "source", "Source utterance", example.source_id) +
    audioCard(example, "target", "Target-speaker reference", example.target_id);
  document.querySelector("#comparison-groups").innerHTML = groups.map((group) => `
    <section class="audio-group">
      <h3>${group.title}</h3>
      <div class="audio-grid">
        ${group.tracks.map(([id, label]) => audioCard(example, id, label, "converted", group.proposed)).join("")}
      </div>
    </section>`).join("");
  bindExclusivePlayback();
}

function bindExclusivePlayback() {
  document.querySelectorAll("audio").forEach((audio) => {
    audio.addEventListener("play", () => {
      document.querySelectorAll("audio").forEach((other) => {
        if (other !== audio) other.pause();
      });
    });
  });
}

function renderExampleTabs() {
  const tabs = document.querySelector("#example-tabs");
  tabs.innerHTML = examples.map((example, index) => `
    <button type="button" role="tab" aria-selected="${index === selectedExample}"
      data-example="${index}">${example.display_name || `Example ${index + 1}`}</button>`).join("");
  tabs.querySelectorAll("[data-example]").forEach((tab) => {
    tab.addEventListener("click", () => {
      selectedExample = Number(tab.dataset.example);
      renderExample();
    });
  });
}

function renderResults() {
  const meta = metricMeta[selectedMetric];
  const maxValue = Math.max(...resultRows.map((row) => row[selectedMetric][0]));
  document.querySelector("#chart-title").textContent = meta.title;
  const direction = document.querySelector("#chart-direction");
  direction.textContent = meta.direction;
  direction.hidden = !meta.direction;
  document.querySelectorAll("[data-metric]").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.metric === selectedMetric));
  });
  document.querySelector("#bar-chart").innerHTML = resultRows.map((row) => {
    const [value, display] = row[selectedMetric];
    return `
      <div class="bar-row bar-${row.group}">
        <span class="bar-label">${row.label}</span>
        <span class="bar-track"><i style="width:${(value / maxValue) * 100}%"></i></span>
        <span class="bar-value">${display}</span>
      </div>`;
  }).join("");
}

document.querySelectorAll("[data-metric]").forEach((tab) => {
  tab.addEventListener("click", () => {
    selectedMetric = tab.dataset.metric;
    renderResults();
  });
});

fetch("audio/manifest.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((data) => {
    examples = data;
    renderExampleTabs();
    renderExample();
  })
  .catch(() => {
    document.querySelector("#pair-label").textContent = "Audio manifest could not be loaded.";
  });

renderResults();
