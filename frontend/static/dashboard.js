/**
 * Trading Dashboard — Client-side logic.
 *
 * Fetches /api/health (for header) and /api/positions (for the table),
 * renders positions via Tabulator with sorting, filtering, and expandable
 * detail rows.  Auto-refreshes every 30 seconds.
 */

// ── State ──

let table = null;
let allPositions = [];
let liveAlpacaData = {};  // keyed by symbol, from /api/live-alpaca
let currentStatusFilter = 'all';
const REFRESH_INTERVAL_MS = 30_000;

// Database tab state
const DB_PAGE_SIZE = 100;
let dbTables = [];
let dbCurrentTable = null;
let dbOffset = 0;
let dbTable = null;

// Friendly labels + badge classes per strategy registry key.
const STRATEGY_LABELS = {
    'rsi_mean_reversion': { label: 'RSI Mean Rev.', cls: 'badge-strategy-rsi' },
    'rvol_orb': { label: 'RVOL ORB', cls: 'badge-strategy-rvol' },
};

function strategyInfo(name) {
    if (!name) return null;
    return STRATEGY_LABELS[name] || { label: name, cls: 'badge-strategy-other' };
}

function strategyBadge(name) {
    const info = strategyInfo(name);
    if (!info) return '<span class="badge badge-strategy-other">—</span>';
    return `<span class="badge ${info.cls}">${info.label}</span>`;
}

// ── DOM refs ──

const $ = (sel) => document.querySelector(sel);

const dom = {
    envBadge: $('#env-badge'),
    paperBadge: $('#paper-badge'),
    statusDot: $('#status-dot'),
    lastRunStatus: $('#last-run-status'),
    lastRunDuration: $('#last-run-duration'),
    backtestCount: $('#backtest-count'),
    lastRefresh: $('#last-refresh'),
    positionCount: $('#position-count'),
    filterBtns: document.querySelectorAll('.filter-btn'),
    // Database tab
    dbTableSelect: $('#db-table-select'),
    dbRefreshBtn: $('#db-refresh-btn'),
    dbRowCount: $('#db-row-count'),
    dbError: $('#db-error'),
    dbPrevBtn: $('#db-prev-btn'),
    dbNextBtn: $('#db-next-btn'),
    dbPageLabel: $('#db-page-label'),
};

// ── Helpers ──

function formatCurrency(val) {
    if (val == null || isNaN(val)) return '—';
    return '$' + Number(val).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatPercent(val) {
    if (val == null || isNaN(val)) return '—';
    return (Number(val) * 100).toFixed(2) + '%';
}

function formatDate(val) {
    if (!val) return '—';
    const d = new Date(val);
    if (isNaN(d.getTime())) return '—';
    return d.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
    });
}

function pnlClass(val) {
    if (val == null) return '';
    return Number(val) >= 0 ? 'pnl-positive' : 'pnl-negative';
}

function sideBadge(side) {
    const cls = side === 'short' ? 'badge badge-short' : 'badge badge-long';
    return `<span class="${cls}">${side || 'long'}</span>`;
}

function formatPositionCount(rows) {
    const base = rows.length + ' position' + (rows.length !== 1 ? 's' : '');
    // Append per-strategy counts of OPEN positions when more than one
    // strategy is represented.
    const byStrategy = {};
    rows.forEach(function (row) {
        if (row.closed) return;
        const name = row.strategy_name || 'rsi_mean_reversion';
        byStrategy[name] = (byStrategy[name] || 0) + 1;
    });
    const names = Object.keys(byStrategy);
    if (names.length > 1) {
        const parts = names.map(function (name) {
            const info = strategyInfo(name);
            const label = info ? info.label : name;
            return label + ' ' + byStrategy[name];
        });
        return base + ' · ' + parts.join(' · ');
    }
    return base;
}

function closedBadge(closed) {
    const cls = closed ? 'badge badge-closed' : 'badge badge-open';
    const text = closed ? 'CLOSED' : 'OPEN';
    return `<span class="${cls}">${text}</span>`;
}

function orderStatusBadge(status) {
    if (!status) return '<span class="badge badge-order-none">—</span>';
    var cls = 'badge-order-unknown';
    var label = status;
    switch (status.toLowerCase()) {
        case 'working':
        case 'new':
        case 'accepted':
        case 'pending_new':
            cls = 'badge-order-working';
            label = 'WORKING';
            break;
        case 'held':
            cls = 'badge-order-held';
            label = 'HELD';
            break;
        case 'filled':
            cls = 'badge-order-filled';
            label = 'FILLED';
            break;
        case 'cancelled':
        case 'canceled':
            cls = 'badge-order-cancelled';
            label = 'CANCEL';
            break;
        case 'rejected':
            cls = 'badge-order-rejected';
            label = 'REJECT';
            break;
        case 'expired':
            cls = 'badge-order-expired';
            label = 'EXPIRED';
            break;
        case 'partially_filled':
            cls = 'badge-order-partial';
            label = 'PARTIAL';
            break;
    }
    return '<span class="badge ' + cls + '">' + label + '</span>';
}


function unrealizedPnl(row) {
    const entry = row.entry_price;
    // Prefer live Alpaca price; fall back to stored current_price
    const current = row._liveCurrentPrice != null ? row._liveCurrentPrice : row.current_price;
    const qty = row.quantity;
    const side = row.side || 'long';
    if (!entry || !current || !qty) return null;
    if (side === 'short') {
        return ((entry - current) / entry) * Math.abs(qty) * current;
    }
    return ((current - entry) / entry) * Math.abs(qty) * current;
}

function unrealizedPnlPct(row) {
    const entry = row.entry_price;
    // Prefer live Alpaca price; fall back to stored current_price
    const current = row._liveCurrentPrice != null ? row._liveCurrentPrice : row.current_price;
    const side = row.side || 'long';
    if (!entry || !current) return null;
    if (side === 'short') {
        return (entry - current) / entry;
    }
    return (current - entry) / entry;
}

// ── Column definitions with superheaders ──
//
// Columns are organized into three column groups:
//   1. Position Record  — fields persisted to CSV / Postgres
//   2. Live (Alpaca)    — real-time market data from the broker
//   3. Unrealized P&L   — computed from the two sources above
//

function groupPositionRecord() {
    return {
        title: '<span class="colgroup-title">Position Record <em>(CSV / DB)</em></span>',
        columns: [
            {
                title: 'Symbol', field: 'symbol',
                sorter: 'string', headerFilter: 'input',
                headerFilterPlaceholder: 'Filter…',
                width: 85,
            },
            {
                title: 'Strategy', field: 'strategy_name',
                formatter: function (cell) { return strategyBadge(cell.getValue()); },
                sorter: 'string', hozAlign: 'center', width: 125,
                headerFilter: 'list',
                headerFilterParams: {
                    values: Object.keys(STRATEGY_LABELS),
                },
                headerFilterEmptyCheck: function (value) { return value === '' || value == null; },
            },
            {
                title: 'Side', field: 'side',
                formatter: function (cell) { return sideBadge(cell.getValue()); },
                sorter: 'string', headerFilter: 'list',
                headerFilterParams: { values: ['long', 'short'] },
                width: 75, hozAlign: 'center',
            },
            {
                title: 'Shares', field: 'quantity',
                sorter: 'number', hozAlign: 'right', width: 80,
                formatter: function (cell) {
                    var v = cell.getValue();
                    if (v == null) return '—';
                    return Math.abs(v).toLocaleString();
                },
            },
            {
                title: 'Entry $', field: 'entry_price',
                sorter: 'number', hozAlign: 'right', width: 100,
                formatter: function (cell) { return formatCurrency(cell.getValue()); },
            },
            {
                title: 'Entry Date', field: 'entry_date',
                sorter: 'string', hozAlign: 'center', width: 105,
                formatter: function (cell) { return formatDate(cell.getValue()); },
            },
            {
                title: 'Alpha', field: 'alpha',
                sorter: 'number', hozAlign: 'right', width: 80,
                formatter: function (cell) {
                    var v = cell.getValue();
                    return v != null ? Number(v).toFixed(4) : '—';
                },
            },
            {
                title: 'Cur. RSI', field: 'current_rsi',
                sorter: 'number', hozAlign: 'center', width: 80,
                formatter: function (cell) {
                    var v = cell.getValue();
                    return v != null ? Number(v).toFixed(1) : '—';
                },
            },
            {
                title: 'RSI Per', field: 'rsi_period',
                sorter: 'number', hozAlign: 'center', width: 75,
                headerFilter: 'list',
                headerFilterParams: { values: [2, 8, 14, 20, 30] },
            },
            {
                title: 'RSI Low', field: 'rsi_lower',
                sorter: 'number', hozAlign: 'center', width: 80,
            },
            {
                title: 'RSI High', field: 'rsi_upper',
                sorter: 'number', hozAlign: 'center', width: 80,
            },
            {
                title: 'Stop Loss', field: 'stop_loss_price',
                sorter: 'number', hozAlign: 'right', width: 105,
                formatter: function (cell) { return formatCurrency(cell.getValue()); },
            },
            {
                title: 'Take Profit', field: 'take_profit_price',
                sorter: 'number', hozAlign: 'right', width: 105,
                formatter: function (cell) { return formatCurrency(cell.getValue()); },
            },
            {
                title: 'Status', field: 'closed',
                formatter: function (cell) { return closedBadge(cell.getValue()); },
                sorter: 'boolean', hozAlign: 'center', width: 85,
                headerFilter: 'list',
                headerFilterParams: {
                    values: [true, false],
                    trueLabel: 'Closed', falseLabel: 'Open',
                },
            },
            {
                title: 'Exit Date', field: 'exit_date',
                sorter: 'string', hozAlign: 'center', width: 105,
                formatter: function (cell) { return formatDate(cell.getValue()); },
            },
            {
                title: 'Exit $', field: 'exit_price',
                sorter: 'number', hozAlign: 'right', width: 100,
                formatter: function (cell) { return formatCurrency(cell.getValue()); },
            },
            {
                title: 'Real. Return', field: 'realized_return',
                sorter: 'number', hozAlign: 'right', width: 100,
                formatter: function (cell) {
                    var v = cell.getValue();
                    if (v == null) return '—';
                    return '<span class="' + pnlClass(v) + '">' + formatPercent(v) + '</span>';
                },
            },
            {
                title: 'Exit Reason', field: 'exit_reason',
                headerFilter: 'input', headerFilterPlaceholder: 'Filter…',
                width: 130,
                formatter: function (cell) { return cell.getValue() || '—'; },
            },
        ],
    };
}

function groupLiveAlpaca() {
    return {
        title: '<span class="colgroup-title colgroup-alpaca">Live <em>(Alpaca API)</em></span>',
        columns: [
            {
                title: 'Current $', field: '_liveCurrentPrice',
                sorter: 'number', hozAlign: 'right', width: 105,
                formatter: function (cell) {
                    var v = cell.getValue();
                    // fall back to stored current_price if live data unavailable
                    if (v == null) {
                        var row = cell.getRow().getData();
                        v = row.current_price;
                    }
                    return formatCurrency(v);
                },
            },
            {
                title: 'SL Status', field: '_slStatus',
                sorter: 'string', hozAlign: 'center', width: 85,
                formatter: function (cell) { return orderStatusBadge(cell.getValue()); },
            },
            {
                title: 'SL Price', field: '_slPrice',
                sorter: 'number', hozAlign: 'right', width: 95,
                formatter: function (cell) { return formatCurrency(cell.getValue()); },
            },
            {
                title: 'TP Status', field: '_tpStatus',
                sorter: 'string', hozAlign: 'center', width: 85,
                formatter: function (cell) { return orderStatusBadge(cell.getValue()); },
            },
            {
                title: 'TP Price', field: '_tpPrice',
                sorter: 'number', hozAlign: 'right', width: 95,
                formatter: function (cell) { return formatCurrency(cell.getValue()); },
            },
        ],
    };
}

function groupUnrealizedPnl() {
    return {
        title: '<span class="colgroup-title colgroup-pnl">Unrealized P&amp;L <em>(computed)</em></span>',
        columns: [
            {
                title: 'P&amp;L $', field: '_unrealizedPnl',
                sorter: 'number', hozAlign: 'right', width: 105,
                formatter: function (cell) {
                    var v = cell.getValue();
                    if (v == null) return '—';
                    return '<span class="' + pnlClass(v) + '">' + formatCurrency(v) + '</span>';
                },
            },
            {
                title: 'P&amp;L %', field: '_unrealizedPnlPct',
                sorter: 'number', hozAlign: 'right', width: 90,
                formatter: function (cell) {
                    var v = cell.getValue();
                    if (v == null) return '—';
                    return '<span class="' + pnlClass(v) + '">' + formatPercent(v) + '</span>';
                },
            },
        ],
    };
}

function buildColumns() {
    return [
        groupPositionRecord(),
        groupLiveAlpaca(),
        groupUnrealizedPnl(),
    ];
}

// ── Build computed columns from raw data ──

function enrichRow(row) {
    row._unrealizedPnl = row.closed ? null : unrealizedPnl(row);
    row._unrealizedPnlPct = row.closed ? null : unrealizedPnlPct(row);
    return row;
}

/**
 * Merge live Alpaca data (from /api/live-alpaca) into each row.
 * Safe to call even when liveAlpacaData is empty — all _live* fields
 * will be null and the column formatters fall back gracefully.
 */
function enrichLiveAlpaca(row) {
    var live = liveAlpacaData[row.symbol];

    // Current price from Alpaca snapshot
    row._liveCurrentPrice = (live && live.current_price != null) ? live.current_price : null;

    // Stop-loss order
    var sl = (live && live.stop_loss_order) ? live.stop_loss_order : null;
    row._slStatus = sl ? sl.status : null;
    row._slPrice  = sl ? sl.stop_price : null;

    // Take-profit order
    var tp = (live && live.take_profit_order) ? live.take_profit_order : null;
    row._tpStatus = tp ? tp.status : null;
    row._tpPrice  = tp ? tp.limit_price : null;

    return row;
}

// ── Tab switching ──

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(function (btn) {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    document.querySelectorAll('.tab-panel').forEach(function (panel) {
        panel.classList.toggle('active', panel.id === 'tab-' + tabName);
    });
    if (tabName === 'database' && dbTables.length === 0) {
        loadDbTables();
    }
}

// ── Database browser ──

function dbErrorMsg(msg) {
    dom.dbError.textContent = msg || '';
}

async function loadDbTables() {
    dom.dbError.textContent = '';
    try {
        const resp = await fetch('/api/db/tables');
        if (resp.status === 501) {
            const data = await resp.json();
            dbErrorMsg(data.error || 'Database browsing not supported by this storage backend.');
            return;
        }
        if (!resp.ok) {
            dbErrorMsg('Failed to load tables (' + resp.status + ')');
            return;
        }
        const data = await resp.json();
        dbTables = data.tables || [];
        dom.dbTableSelect.innerHTML = '';
        if (dbTables.length === 0) {
            dbErrorMsg('No browsable tables found.');
            return;
        }
        dbTables.forEach(function (t) {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t;
            dom.dbTableSelect.appendChild(opt);
        });
        const saved = dbCurrentTable;
        if (saved && dbTables.indexOf(saved) !== -1) {
            dom.dbTableSelect.value = saved;
        } else {
            dbCurrentTable = dbTables[0];
        }
        await loadDbTable(dbCurrentTable, 0);
    } catch (err) {
        console.error('Failed to load DB tables:', err);
        dbErrorMsg('Failed to load tables.');
    }
}

async function loadDbTable(name, offset) {
    dom.dbError.textContent = '';
    try {
        const resp = await fetch(
            '/api/db/table/' + encodeURIComponent(name) +
            '?limit=' + DB_PAGE_SIZE + '&offset=' + offset
        );
        if (resp.status === 501) {
            const data = await resp.json();
            dbErrorMsg(data.error || 'Database browsing not supported by this storage backend.');
            return;
        }
        if (!resp.ok) {
            const data = await resp.json().catch(function () { return {}; });
            dbErrorMsg(data.error || 'Failed to fetch table (' + resp.status + ')');
            return;
        }
        const data = await resp.json();
        dbCurrentTable = name;
        dbOffset = offset;

        const rows = data.rows || [];
        const total = data.total || 0;
        const cols = data.columns || [];

        dom.dbRowCount.textContent = total + ' row' + (total !== 1 ? 's' : '');
        const pageStart = total === 0 ? 0 : offset + 1;
        const pageEnd = Math.min(offset + rows.length, total);
        dom.dbPageLabel.textContent = rows.length ? (pageStart + '–' + pageEnd + ' of ' + total) : '—';
        dom.dbPrevBtn.disabled = offset <= 0;
        dom.dbNextBtn.disabled = offset + rows.length >= total;

        const columns = cols.map(function (c) { return { title: c, field: c }; });
        if (dbTable) {
            dbTable.setColumns(columns);
            await dbTable.replaceData(rows);
        } else {
            dbTable = new Tabulator('#db-table-container', {
                data: rows,
                columns: columns,
                layout: 'fitDataFill',
                height: 'calc(100vh - 280px)',
                selectable: false,
                columnHeaderVertAlign: 'bottom',
            });
        }
    } catch (err) {
        console.error('Failed to fetch DB table:', err);
        dbErrorMsg('Failed to fetch table.');
    }
}

// ── Fetch & render ──

async function fetchHealth() {
    try {
        const resp = await fetch('/health');
        const data = await resp.json();

        // Environment
        const env = data.environment || 'dev';
        dom.envBadge.textContent = env;
        dom.envBadge.className = 'env-badge env-' + env;

        // Paper trade
        const paper = data.paper_trade !== false;
        dom.paperBadge.textContent = paper ? 'PAPER' : 'LIVE';
        dom.paperBadge.className = paper ? 'paper-badge' : 'paper-badge live';

        // Status
        const status = data.last_run_status || 'unknown';
        dom.statusDot.className = 'status-dot ' + status;
        dom.lastRunStatus.textContent = status.replace(/_/g, ' ');

        // Duration
        const dur = data.last_run_duration_seconds;
        if (dur) {
            const mins = Math.floor(dur / 60);
            const secs = Math.round(dur % 60);
            dom.lastRunDuration.textContent = `${mins}m ${secs}s`;
        } else {
            dom.lastRunDuration.textContent = '—';
        }

        // Backtest count
        dom.backtestCount.textContent = data.last_run_backtest_count ?? '—';
    } catch (err) {
        console.error('Failed to fetch /health:', err);
    }
}

async function fetchLiveAlpaca() {
    try {
        const resp = await fetch('/api/live-alpaca');
        if (!resp.ok) {
            console.error('Live Alpaca fetch failed:', resp.status);
            liveAlpacaData = {};
            return;
        }
        liveAlpacaData = await resp.json();
        if (!liveAlpacaData || typeof liveAlpacaData !== 'object') {
            liveAlpacaData = {};
        }
    } catch (err) {
        console.error('Failed to fetch /api/live-alpaca:', err);
        liveAlpacaData = {};
    }
}

async function fetchPositions() {
    try {
        // 1. Fetch live Alpaca data FIRST (snapshot prices + bracket orders)
        await fetchLiveAlpaca();

        // 2. Fetch position records from storage
        const resp = await fetch('/api/positions');

        // Always show last-refresh timestamp after attempting fetch
        dom.lastRefresh.textContent = new Date().toLocaleTimeString();

        if (!resp.ok) {
            console.error('Positions fetch failed:', resp.status);
            return;
        }
        const data = await resp.json();

        if (!Array.isArray(data)) {
            console.error('Unexpected positions response:', data);
            return;
        }

        // Store the full dataset so filter buttons can slice it client-side
        allPositions = data.map(enrichRow).map(enrichLiveAlpaca);

        // Recompute unrealized P&L now that live prices are merged
        allPositions.forEach(function (row) {
            row._unrealizedPnl = row.closed ? null : unrealizedPnl(row);
            row._unrealizedPnlPct = row.closed ? null : unrealizedPnlPct(row);
        });

        // Apply the current filter
        var filtered = allPositions;
        if (currentStatusFilter === 'open') {
            filtered = allPositions.filter(function (row) { return !row.closed; });
        } else if (currentStatusFilter === 'closed') {
            filtered = allPositions.filter(function (row) { return row.closed; });
        }

        dom.positionCount.textContent = formatPositionCount(filtered);

        if (table) {
            await table.replaceData(filtered);
        } else {
            table = new Tabulator('#positions-table', {
                data: filtered,
                columns: buildColumns(),
                layout: 'fitDataFill',
                height: 'calc(100vh - 155px)',
                selectable: false,
                columnHeaderVertAlign: 'bottom',
            });
        }
    } catch (err) {
        console.error('Failed to fetch /api/positions:', err);
        dom.lastRefresh.textContent = new Date().toLocaleTimeString();
    }
}

// ── Filter buttons ──

dom.filterBtns.forEach((btn) => {
    btn.addEventListener('click', function () {
        dom.filterBtns.forEach((b) => b.classList.remove('active'));
        this.classList.add('active');
        currentStatusFilter = this.dataset.status;

        // Filter client-side — no re-fetch needed
        var filtered = allPositions;
        if (currentStatusFilter === 'open') {
            filtered = allPositions.filter(function (row) { return !row.closed; });
        } else if (currentStatusFilter === 'closed') {
            filtered = allPositions.filter(function (row) { return row.closed; });
        }
        dom.positionCount.textContent = formatPositionCount(filtered);
        if (table) {
            // Re-enrich live data in case it changed between fetches
            filtered.forEach(enrichLiveAlpaca);
            table.replaceData(filtered);
        }
    });
});

// ── Tab events ──

document.querySelectorAll('.tab-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
        switchTab(this.dataset.tab);
    });
});

dom.dbTableSelect.addEventListener('change', function () {
    dbOffset = 0;
    loadDbTable(this.value, 0);
});
dom.dbRefreshBtn.addEventListener('click', function () {
    if (dbCurrentTable) loadDbTable(dbCurrentTable, dbOffset);
});
dom.dbPrevBtn.addEventListener('click', function () {
    if (dbCurrentTable) loadDbTable(dbCurrentTable, Math.max(0, dbOffset - DB_PAGE_SIZE));
});
dom.dbNextBtn.addEventListener('click', function () {
    if (dbCurrentTable) loadDbTable(dbCurrentTable, dbOffset + DB_PAGE_SIZE);
});

// ── Init ──

async function init() {
    await fetchHealth();
    await fetchPositions();

    // Auto-refresh positions (includes live Alpaca data internally)
    setInterval(fetchPositions, REFRESH_INTERVAL_MS);
    // Refresh header less frequently
    setInterval(fetchHealth, REFRESH_INTERVAL_MS * 2);
}

// ── Run Now button ──

function setupRunNowButton() {
    var btn = $('#run-now-btn');
    if (!btn) return;

    btn.disabled = false;
    btn.addEventListener('click', async function () {
        if (btn.disabled) return;
        btn.disabled = true;
        btn.textContent = '⏳ Running...';

        try {
            var resp = await fetch('/api/run-cycle', { method: 'POST' });
            var data = await resp.json();
            if (resp.ok) {
                btn.textContent = '✅ Triggered';
                setTimeout(function () {
                    btn.textContent = '▶ Run Now';
                    btn.disabled = false;
                }, 3000);
            } else if (resp.status === 409) {
                // Already running
                btn.textContent = '⏳ Already running';
                setTimeout(function () {
                    btn.textContent = '▶ Run Now';
                    btn.disabled = false;
                }, 5000);
            } else {
                btn.textContent = '❌ Error';
                console.error('Run cycle failed:', data);
                setTimeout(function () {
                    btn.textContent = '▶ Run Now';
                    btn.disabled = false;
                }, 3000);
            }
        } catch (err) {
            console.error('Run cycle request failed:', err);
            btn.textContent = '❌ Error';
            setTimeout(function () {
                btn.textContent = '▶ Run Now';
                btn.disabled = false;
            }, 3000);
        }
    });
}

document.addEventListener('DOMContentLoaded', function () {
    init();
    setupRunNowButton();
});
