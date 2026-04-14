const API = '';
const CHART_SRCS = {
  loss:      '/static/plots/loss_curve.png',
  mae:       '/static/plots/mae_curve.png',
  scatter:   '/static/plots/actual_vs_pred.png',
};

const $ = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);

// Dynamic theme mapping based on IPL teams
const TEAM_THEMES = {
  "Mumbai Indians": "theme-mi",
  "Chennai Super Kings": "theme-csk",
  "Royal Challengers Bangalore": "theme-rcb",
  "Kolkata Knight Riders": "theme-kkr",
  "Delhi Capitals": "theme-dc",
  "Sunrisers Hyderabad": "theme-srh",
  "Rajasthan Royals": "theme-rr",
  "Punjab Kings": "theme-pk",
  "Lucknow Super Giants": "theme-lsg",
  "Gujarat Titans": "theme-gt"
};

// Application State
let appState = {
  isPredicting: false,
  metrics: null,
  options: null,
  chartData: null,
  activeChart: null
};

// DOM Init
document.addEventListener("DOMContentLoaded", initApp);

async function initApp() {
  bindUIEvents();
  
  try {
    const [optRes, metRes, chartRes] = await Promise.all([
      fetch(`${API}/api/options`),
      fetch(`${API}/api/metrics`),
      fetch(`${API}/api/charts`)
    ]);

    if (!optRes.ok || !metRes.ok || !chartRes.ok) throw new Error('API Sync Failed');

    appState.options = await optRes.json();
    appState.metrics = await metRes.json();
    appState.chartData = await chartRes.json();

    populateSelects();
    updateTopMetrics();
    initChart(); // Initialize Chart.js
    setConnectionStatus('online', 'ENGINE ONLINE');
    
    console.log('[DL Engine] Boot Sequence Complete.');
  } catch(e) {
    console.warn('[Offline Mode] Using local simulation cache', e);
    setConnectionStatus('offline', 'OFFLINE SIM');
    seedDemoOptions();
  }
}

function initChart() {
  const ctx = document.getElementById('realtimeChart').getContext('2d');
  
  const initialData = appState.chartData.venue_avg;
  
  appState.activeChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: initialData.labels.slice(0, 10), // Limit to top 10 for compactness
      datasets: [{
        label: 'Average Score',
        data: initialData.data.slice(0, 10),
        backgroundColor: 'rgba(45, 108, 223, 0.8)',
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { boxPadding: 6 }
      },
      scales: {
        y: { beginAtZero: false, min: 140 } // scores usually don't drop to 0
      }
    }
  });
}

function updateChart(datasetKey) {
  if(!appState.activeChart || !appState.chartData) return;
  const data = appState.chartData[datasetKey];
  
  let type = 'bar';
  let color = 'rgba(45, 108, 223, 0.8)';
  if (datasetKey === 'wickets_score') {
      type = 'line';
      color = 'rgba(255, 51, 102, 0.8)';
      appState.activeChart.options.scales.y.min = 0;
  } else {
      appState.activeChart.options.scales.y.min = 140; 
  }

  appState.activeChart.config.type = type;
  appState.activeChart.data.labels = data.labels.slice(0, 10);
  appState.activeChart.data.datasets[0].data = data.data.slice(0, 10);
  appState.activeChart.data.datasets[0].backgroundColor = color;
  appState.activeChart.data.datasets[0].borderColor = color;
  
  appState.activeChart.update();
}

function setConnectionStatus(status, text) {
  const dot = $('#server-status');
  const txt = $('#server-text');
  dot.className = `status-dot ${status}`;
  txt.textContent = text;
  if(status === 'online') txt.style.color = '#10b981';
}

function populateSelects() {
  const opt = appState.options;
  fillOptions('#batting_team', opt.batting_team);
  fillOptions('#bowling_team', opt.bowling_team);
  fillOptions('#venue', opt.venue);
  
  // Disable until team is selected
  $('#batsman').innerHTML = '<option value="" disabled selected>Select Batting Team First</option>';
  $('#bowler').innerHTML = '<option value="" disabled selected>Select Bowling Team First</option>';
}

function fillOptions(selector, items) {
  const select = $(selector);
  select.innerHTML = '<option value="" disabled selected>Select Parameter...</option>';
  if(!items) return;
  items.sort().forEach(val => {
    let el = document.createElement('option');
    el.value = el.textContent = val;
    select.appendChild(el);
  });
}

function updateTopMetrics() {
  if(!appState.metrics) return;
  $('#m-r2').textContent = appState.metrics.R2 ? (appState.metrics.R2 * 100).toFixed(1) + '%' : '--';
  $('#m-mae').textContent = appState.metrics.MAE ? '±' + appState.metrics.MAE.toFixed(1) : '--';
}

function bindUIEvents() {
  // Dynamic Dropdowns: Batting Team -> Batsmen
  $('#batting_team').addEventListener('change', e => {
    const team = e.target.value;
    document.body.className = TEAM_THEMES[team] || 'theme-default';
    if(team) $('#flag-team').textContent = team;
    
    // Fill batsmen for this team
    if(appState.options && appState.options.team_batsmen[team]) {
        fillOptions('#batsman', appState.options.team_batsmen[team]);
    }
  });

  // Dynamic Dropdowns: Bowling Team -> Bowlers
  $('#bowling_team').addEventListener('change', e => {
    const team = e.target.value;
    if(appState.options && appState.options.team_bowlers[team]) {
        fillOptions('#bowler', appState.options.team_bowlers[team]);
    }
  });

  // Player Graphs Integration
  $('#batsman').addEventListener('change', e => fetchPlayerAnalysis(e.target.value, 'batsman'));
  $('#bowler').addEventListener('change', e => fetchPlayerAnalysis(e.target.value, 'bowler'));


  const inputsToTrack = ['#current_score', '#over', '#ball', '#runs_last_5_overs'];
  inputsToTrack.forEach(sel => {
    $(sel).addEventListener('input', calculateLiveStats);
  });

  // Analytics Tabs
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      $$('.tab-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      
      $('#player-stats-box').style.display = 'none'; // reset
      
      const targetChart = e.target.dataset.target;
      if (targetChart === 'venue') updateChart('venue_avg');
      if (targetChart === 'team') updateChart('team_avg');
      if (targetChart === 'wickets') updateChart('wickets_score');
    });
  });

  // Prediction Submit
  $('#prediction-form').addEventListener('submit', handlePrediction);
}

async function fetchPlayerAnalysis(playerName, role) {
  if(!playerName) return;
  
  try {
    const res = await fetch(`${API}/api/player_stats?player=${encodeURIComponent(playerName)}&role=${role}`);
    const data = await res.json();
    if(data.error) return;
    
    // Show stats box
    const statBox = $('#player-stats-box');
    statBox.style.display = 'flex';
    statBox.innerHTML = '';
    
    for (const [key, value] of Object.entries(data.stats)) {
      statBox.innerHTML += `
        <div style="text-align: center;">
          <small style="color:var(--text-muted); font-size:0.7rem; text-transform:uppercase; font-weight:700;">${key}</small>
          <div style="font-size:1.2rem; font-weight:900; color:var(--text-dark);">${value}</div>
        </div>
      `;
    }
    
    // Update chart to show player analysis
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    
    let chartColor = role === 'batsman' ? 'rgba(0, 240, 255, 0.8)' : 'rgba(255, 51, 102, 0.8)';
    
    appState.activeChart.config.type = 'bar';
    appState.activeChart.data.labels = data.labels;
    appState.activeChart.data.datasets[0].data = data.data;
    appState.activeChart.data.datasets[0].backgroundColor = chartColor;
    appState.activeChart.data.datasets[0].borderColor = chartColor;
    appState.activeChart.data.datasets[0].label = `Score Frequency for ${playerName}`;
    appState.activeChart.options.scales.y.min = 0;
    appState.activeChart.options.plugins.legend.display = true;
    appState.activeChart.update();
  } catch(e) {
    console.error("Failed to load player analysis", e);
  }
}


function calculateLiveStats() {
  const score = parseInt($('#current_score').value) || 0;
  const overs = parseInt($('#over').value) || 0;
  const balls = parseInt($('#ball').value) || 0;
  const last5 = parseInt($('#runs_last_5_overs').value) || 0;

  // CRR
  const totalOvers = overs + (balls / 6);
  const crr = totalOvers > 0 ? (score / totalOvers).toFixed(2) : "0.00";
  $('#live-crr').textContent = crr;

  // Momentum Bar (Rough estimate: max 100 runs in 5 overs is 100% momentum)
  const momentumPercent = Math.min(100, (last5 / 100) * 100);
  $('#momentum-fill').style.width = `${momentumPercent}%`;
}

async function handlePrediction(e) {
  e.preventDefault();
  if(appState.isPredicting) return;

  appState.isPredicting = true;
  const btn = $('#predict-btn');
  btn.classList.add('processing');
  $('.btn-content').textContent = 'CALCULATING VARIANCES...';

  // Toggle UI State
  switchResultState('result-calc');

  const payload = {
    batting_team: $('#batting_team').value,
    bowling_team: $('#bowling_team').value,
    venue: $('#venue').value,
    batsman: $('#batsman').value,
    bowler: $('#bowler').value,
    over: parseInt($('#over').value),
    ball: parseInt($('#ball').value),
    current_score: parseInt($('#current_score').value),
    wickets: parseInt($('#wickets').value),
    runs_last_5_overs: parseInt($('#runs_last_5_overs').value),
    wickets_last_5_overs: parseInt($('#wickets_last_5_overs').value)
  };

  try {
    // Artificial delay to show calculation animation
    await new Promise(r => setTimeout(r, 1200));

    const res = await fetch(`${API}/api/predict`, {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(payload),
    });

    if(!res.ok) throw new Error('Prediction API Error');
    const data = await res.json();
    
    renderResults(data, payload);
    switchResultState('result-done');

  } catch(err) {
    console.error(err);
    // Offline Demo fallback
    const est = runDemoFallback(payload);
    renderResults(est, payload);
    switchResultState('result-done');
  } finally {
    appState.isPredicting = false;
    btn.classList.remove('processing');
    $('.btn-content').textContent = 'RE-INITIALIZE PROJECTION';
  }
}

function renderResults(data, payload) {
  // Animate Odometer
  const scoreEl = $('#final-score');
  const targetScore = data.predicted_score;
  animateOdometer(scoreEl, parseInt(payload.current_score), targetScore, 1000);

  $('#score-range').textContent = `${data.range_low} – ${data.range_high}`;
  $('#conf-val').textContent = `${(data.confidence || 85.5).toFixed(1)}%`;

  // Required RR
  const oversLeft = 20 - (payload.over + (payload.ball/6));
  const runsReq = targetScore - payload.current_score;
  if(oversLeft > 0 && runsReq > 0) {
    $('#req-rr').textContent = (runsReq / oversLeft).toFixed(2);
  } else {
    $('#req-rr').textContent = 'N/A';
  }
  
  // Highlight Result Card
  $('.scanning-beam').style.opacity = '1';
  setTimeout(() => $('.scanning-beam').style.opacity = '0', 2000);
}

function switchResultState(stateId) {
  $$('.result-state').forEach(el => el.classList.remove('active'));
  $(`#${stateId}`).classList.add('active');
}

function animateOdometer(element, start, end, duration) {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    // Easing Out Quartic
    const ease = 1 - Math.pow(1 - progress, 4);
    const text = Math.floor(ease * (end - start) + start);
    element.innerHTML = text;
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}

// Minimal Fallback if Python backend is offline
function runDemoFallback(p) {
  const rr = p.current_score / (p.over || 10);
  const est = Math.round(p.current_score + rr * (20 - p.over) + (Math.random()*10 - 5));
  return {
    predicted_score: Math.max(est, p.current_score),
    range_low: est - 10,
    range_high: est + 10,
    confidence: 76.8
  };
}

function seedDemoOptions() {
  const teams = ["Mumbai Indians","Chennai Super Kings","Royal Challengers Bangalore","Kolkata Knight Riders","Delhi Capitals","Sunrisers Hyderabad","Rajasthan Royals","Punjab Kings","Lucknow Super Giants","Gujarat Titans"];
  fillOptions('#batting_team', teams); fillOptions('#bowling_team', teams);
  fillOptions('#venue', ["Wankhede Stadium","MA Chidambaram Stadium","Eden Gardens"]);
  fillOptions('#batsman', ["Rohit Sharma","Virat Kohli","MS Dhoni"]);
  fillOptions('#bowler', ["Jasprit Bumrah","Rashid Khan","Yuzvendra Chahal"]);
  $('#m-r2').textContent = "86.1%"; $('#m-mae').textContent = "±12.1";
}
