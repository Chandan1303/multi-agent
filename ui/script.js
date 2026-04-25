let chart1 = null;
let chart2 = null;

async function runSimulation() {
    const btn = document.getElementById('run-btn');
    btn.disabled = true;
    btn.textContent = '⏳ Running...';

    // Clear previous table
    document.getElementById('log-tbody').innerHTML = '';
    ['stat-pollution','stat-economy','stat-satisfaction','stat-reward'].forEach(id => {
        document.getElementById(id).textContent = '--';
    });

    try {
        const res = await fetch('/simulate', { method: 'POST' });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        updateStats(data);
        renderCharts(data);
        updateTable(data);
    } catch (err) {
        console.error("Simulation error:", err);
        document.getElementById('stat-pollution').innerHTML =
            "<span style='font-size:1rem;color:#ef4444'>Error: " + err.message + "</span>";
    } finally {
        btn.disabled = false;
        btn.textContent = '▶ Run Simulation';
    }
}

function updateStats(data) {
    const last = data[data.length - 1];
    document.getElementById('stat-pollution').textContent = last.pollution.toFixed(1);
    document.getElementById('stat-economy').textContent = last.economy.toFixed(1);
    document.getElementById('stat-satisfaction').textContent = last.satisfaction.toFixed(1);
    const rewardEl = document.getElementById('stat-reward');
    rewardEl.textContent = last.reward.toFixed(1);
    rewardEl.style.color = last.reward > 0 ? '#10b981' : '#ef4444';
}

function renderCharts(data) {
    const labels = data.map(d => `Day ${d.day}`);
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    if (chart1) chart1.destroy();
    if (chart2) chart2.destroy();

    chart1 = new Chart(document.getElementById('pollutionRewardChart').getContext('2d'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'Pollution Level', data: data.map(d => d.pollution), borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', borderWidth: 3, fill: true, tension: 0.4 },
                { label: 'Calculated Reward', data: data.map(d => d.reward), borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', borderWidth: 3, fill: true, tension: 0.4 }
            ]
        },
        options: { responsive: true, interaction: { mode: 'index', intersect: false }, plugins: { legend: { position: 'top' } }, scales: { y: { grid: { color: 'rgba(255,255,255,0.05)' } }, x: { grid: { display: false } } } }
    });

    chart2 = new Chart(document.getElementById('economySatisfactionChart').getContext('2d'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'Economy', data: data.map(d => d.economy), borderColor: '#38bdf8', borderWidth: 3, tension: 0.4 },
                { label: 'Satisfaction', data: data.map(d => d.satisfaction), borderColor: '#c084fc', borderWidth: 3, tension: 0.4 }
            ]
        },
        options: { responsive: true, interaction: { mode: 'index', intersect: false }, plugins: { legend: { position: 'top' } }, scales: { y: { grid: { color: 'rgba(255,255,255,0.05)' } }, x: { grid: { display: false } } } }
    });
}

function updateTable(data) {
    const tbody = document.getElementById('log-tbody');
    data.slice(-10).reverse().forEach(d => {
        const tr = document.createElement('tr');
        const weatherIcon = d.weather === 'sunny' ? '☀️' : '🌧️';
        const indAct = d.actions.industry_action.replace(/_/g, ' ');
        const govAct = d.actions.government_action.replace(/_/g, ' ');
        tr.innerHTML = `
            <td><strong>Day ${d.day}</strong></td>
            <td>${weatherIcon} <span style="text-transform:capitalize;margin-left:5px">${d.weather}</span></td>
            <td>
                <div style="display:flex;align-items:center;gap:10px">
                    <div style="background:rgba(255,255,255,0.1);width:80px;height:8px;border-radius:4px;overflow:hidden">
                        <div style="background:${d.pollution > 60 ? '#ef4444' : '#10b981'};width:${d.pollution}%;height:100%"></div>
                    </div>
                    <span style="font-weight:600">${d.pollution.toFixed(1)}</span>
                </div>
            </td>
            <td><span style="font-size:0.85rem;background:rgba(56,189,248,0.15);padding:4px 10px;border-radius:20px;color:#38bdf8;text-transform:capitalize">${indAct}</span></td>
            <td><span style="font-size:0.85rem;background:rgba(192,132,252,0.15);padding:4px 10px;border-radius:20px;color:#c084fc;text-transform:capitalize">${govAct}</span></td>
        `;
        tbody.appendChild(tr);
    });
}
