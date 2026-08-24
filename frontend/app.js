'use strict';

/* ============================================================
 * Risk Modeling App - Frontend JavaScript
 * ============================================================
 * Workflow: Upload -> EDA -> Binning -> Training ->
 *           Evaluation -> Copilot -> Export -> Deploy -> Monitor -> Explain
 * Backend API: http://localhost:8080
 * ============================================================ */

/* ============================================================
 * Configuration
 * ============================================================ */

// Allow overriding the backend API base via ?api=https://your-host:8080
const __urlParams = new URLSearchParams(window.location.search);
const __apiOverride = __urlParams.get('api');

const CONFIG = {
    apiBase: __apiOverride || 'local',
    apiPrefix: '/api/v1',
    defaultScreen: 'upload',
    requestTimeout: 120000,
};

// Ensure global App object exists early (will be populated later)
window.App = window.App || {};

const SCREENS = [
    'upload', 'eda', 'binning', 'training', 'evaluation',
    'copilot', 'export', 'deploy', 'monitor', 'explain',
];

/* ============================================================
 * Global State
 * ============================================================ */

const state = {
    currentScreen: null,
    apiOnline: false,
    dataSummary: null,
    dataPreview: null,
    edaResult: null,
    binningResult: null,
    selectedFeatures: null,
    selectedFeature: null,
    trainResult: null,
    evalResult: null,
    scorecard: null,
    ivRanking: null,
    targetCol: null,
};

/* ============================================================
 * API Helper Class
 * ============================================================ */

class RiskModelAPI {
    /**
     * Wrapper around fetch for all backend API calls.
     * Handles JSON parsing, error extraction, and loading state.
     * @param {string} baseURL - Base URL of the backend API
     */
    constructor(baseURL) {
        this.baseURL = baseURL || CONFIG.apiBase;
    }

    /**
     * Build a full URL from a relative path.
     * @param {string} url - Relative or absolute URL
     * @returns {string}
     */
    _buildURL(url) {
        if (url.startsWith('http://') || url.startsWith('https://')) return url;
        return this.baseURL + url;
    }

    /**
     * Core fetch wrapper with error handling.
     * @param {string} url - Request URL
     * @param {object} options - Fetch options
     * @param {boolean} silent - Skip loading overlay
     * @returns {Promise<Response>}
     */
    async _request(url, options = {}, silent = false) {
        if (!silent) showLoading();
        try {
            // Pure-frontend mode: route to the embedded JS engine (no backend needed).
            if (this.baseURL === 'local' && typeof window.RiskEngineLocal !== 'undefined') {
                const method = options.method || 'GET';
                let body = null;
                if (options.body) {
                    body = typeof options.body === 'string' ? JSON.parse(options.body) : options.body;
                }
                return await window.RiskEngineLocal.handle(method, url, body);
            }
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), CONFIG.requestTimeout);
            options.signal = controller.signal;

            const response = await fetch(this._buildURL(url), options);
            clearTimeout(timer);

            if (!response.ok) {
                let detail = `HTTP ${response.status} ${response.statusText}`;
                try {
                    const body = await response.json();
                    detail = body.detail || body.message || body.error || JSON.stringify(body);
                } catch {
                    try {
                        const text = await response.text();
                        if (text) detail = text;
                    } catch { /* ignore */ }
                }
                throw new ApiError(detail, response.status);
            }
            return response;
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new ApiError('请求超时，请检查网络连接或稍后重试', 408);
            }
            if (error instanceof ApiError) throw error;
            throw new ApiError(`网络请求失败: ${error.message}`, 0);
        } finally {
            if (!silent) hideLoading();
        }
    }

    /**
     * GET request returning parsed JSON (or text for non-JSON responses).
     * @param {string} url - API endpoint path
     * @param {object} opts - { silent: boolean }
     * @returns {Promise<any>}
     */
    async get(url, opts = {}) {
        const silent = !!opts.silent;
        const response = await this._request(url, { method: 'GET' }, silent);
        return this._parseBody(response);
    }

    /**
     * POST request with JSON body, returning parsed response.
     * @param {string} url - API endpoint path
     * @param {object} data - Request body (serialized as JSON)
     * @param {object} opts - { silent: boolean }
     * @returns {Promise<any>}
     */
    async post(url, data = {}, opts = {}) {
        const silent = !!opts.silent;
        const response = await this._request(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        }, silent);
        return this._parseBody(response);
    }

    /**
     * POST request with FormData for file uploads.
     * @param {string} url - API endpoint path
     * @param {FormData} formData - Form data object
     * @param {object} opts - { silent: boolean }
     * @returns {Promise<any>}
     */
    async upload(url, formData, opts = {}) {
        const silent = !!opts.silent;
        const response = await this._request(url, {
            method: 'POST',
            body: formData,
        }, silent);
        return this._parseBody(response);
    }

    /**
     * Parse response body based on Content-Type.
     * @param {Response} response
     * @returns {Promise<any>}
     */
    async _parseBody(response) {
        const ct = response.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
            return await response.json();
        }
        return await response.text();
    }

    /**
     * GET request returning raw text (for HTML / file downloads).
     * @param {string} url
     * @returns {Promise<string>}
     */
    async getText(url) {
        const response = await this._request(url, { method: 'GET' }, false);
        return await response.text();
    }

    /**
     * POST request returning raw text (for export endpoints).
     * @param {string} url
     * @param {object} data
     * @returns {Promise<string>}
     */
    async postText(url, data = {}) {
        const response = await this._request(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        }, false);
        return await response.text();
    }

    /**
     * POST request returning a Blob (for file downloads).
     * @param {string} url
     * @param {object} data
     * @returns {Promise<Blob>}
     */
    async postBlob(url, data = {}) {
        const response = await this._request(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        }, false);
        return await response.blob();
    }

    /**
     * Check API health endpoint.
     * @returns {Promise<boolean>}
     */
    async checkHealth() {
        try {
            const result = await this.get('/api/v1/health', { silent: true });
            this.online = true;
            updateAPIStatus(true);
            return result.status === 'ok';
        } catch {
            this.online = false;
            updateAPIStatus(false);
            return false;
        }
    }
}

/**
 * Custom error class for API errors.
 */
class ApiError extends Error {
    constructor(message, statusCode) {
        super(message);
        this.name = 'ApiError';
        this.statusCode = statusCode;
    }
}

/* ============================================================
 * Utility Functions
 * ============================================================ */

/**
 * Escape HTML to prevent XSS injection.
 * @param {string} str
 * @returns {string}
 */
function escapeHTML(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/**
 * Show a toast notification.
 * @param {string} message - Message text
 * @param {string} type - success | error | warning | info
 * @param {number} duration - Display duration in ms
 */
function showToast(message, type = 'info', duration = 3500) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = { success: '\u2713', error: '\u2717', warning: '\u26A0', info: '\u2139' };
    toast.innerHTML = `<span class="toast-icon">${icon[type] || icon.info}</span><span class="toast-msg">${escapeHTML(message)}</span>`;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('toast-show'));
    setTimeout(() => {
        toast.classList.remove('toast-show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/* --- Loading overlay management (ref-counted for concurrent requests) --- */

let _loadingCount = 0;

/**
 * Show the loading overlay (ref-counted).
 */
function showLoading() {
    _loadingCount++;
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.style.display = 'flex';
}

/**
 * Hide the loading overlay (ref-counted).
 */
function hideLoading() {
    _loadingCount = Math.max(0, _loadingCount - 1);
    if (_loadingCount === 0) {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.style.display = 'none';
    }
}

/**
 * Format a number with fixed decimal places.
 * @param {number|string|null} n
 * @param {number} decimals
 * @returns {string}
 */
function formatNumber(n, decimals = 2) {
    if (n === null || n === undefined || n === '' || isNaN(Number(n))) return '-';
    return Number(n).toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });
}

/**
 * Format a fraction as a percentage string.
 * @param {number} n - Fraction (0.3 => "30.00%")
 * @param {number} decimals
 * @returns {string}
 */
function formatPercent(n, decimals = 2) {
    if (n === null || n === undefined || isNaN(Number(n))) return '-';
    return (Number(n) * 100).toFixed(decimals) + '%';
}

/**
 * Return a color based on a credit score value.
 * @param {number} score
 * @returns {string} Hex color
 */
function colorScore(score) {
    if (score >= 660) return '#10b981';  // green - low risk
    if (score >= 580) return '#3b82f6';  // blue  - medium risk
    if (score >= 500) return '#f59e0b';  // amber - high risk
    return '#ef4444';                    // red   - very high risk
}

/**
 * Return a color based on WOE value.
 * @param {number} woe
 * @returns {string}
 */
function colorWOE(woe) {
    if (woe > 0.1) return '#10b981';
    if (woe < -0.1) return '#ef4444';
    return '#6b7280';
}

/**
 * Return a color based on IV strength.
 * @param {number} iv
 * @returns {string}
 */
function colorIV(iv) {
    if (iv >= 0.3) return '#10b981';  // strong
    if (iv >= 0.1) return '#f59e0b';  // medium
    return '#6b7280';                  // weak
}

/**
 * Return a color based on PSI value.
 * @param {number} psi
 * @returns {string}
 */
function colorPSI(psi) {
    if (psi < 0.1) return '#10b981';   // stable
    if (psi < 0.25) return '#f59e0b';   // warning
    return '#ef4444';                   // unstable
}

/**
 * Return a background color for a correlation value (-1 to 1).
 * @param {number} val
 * @returns {string}
 */
function correlationColor(val) {
    const alpha = Math.min(Math.abs(val), 1);
    if (val >= 0) return `rgba(16, 185, 129, ${alpha})`;  // green
    return `rgba(239, 68, 68, ${alpha})`;                   // red
}

/**
 * Generate HTML for a horizontal bar.
 * @param {number} value - Current value
 * @param {number} max - Maximum value (for width scaling)
 * @param {string} color - CSS color
 * @returns {string} HTML string
 */
function renderBar(value, max, color = '#3b82f6') {
    const safeMax = max > 0 ? max : 1;
    const width = Math.min(100, Math.max(0, (Math.abs(value) / safeMax) * 100));
    return `<div class="bar-wrapper">`
        + `<div class="bar-track">`
        + `<div class="bar-fill" style="width:${width}%;background-color:${color};"></div>`
        + `</div>`
        + `<span class="bar-value">${formatNumber(value, 4)}</span>`
        + `</div>`;
}

/**
 * Create a DOM element with optional class and text content.
 * @param {string} tag - HTML tag name
 * @param {string} className - CSS class
 * @param {string} textContent
 * @returns {HTMLElement}
 */
function createElement(tag, className, textContent) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (textContent !== undefined) el.textContent = textContent;
    return el;
}

/**
 * Clear the innerHTML of a container by ID.
 * @param {string} id - Element ID
 */
function clearContainer(id) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '';
}

/**
 * Set the innerHTML of a container by ID.
 * @param {string} id - Element ID
 * @param {string} html - HTML content
 */
function setContainerHTML(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
}

/**
 * Update the API status indicator in the UI.
 * @param {boolean} isOnline
 */
function updateAPIStatus(isOnline) {
    state.apiOnline = isOnline;
    const indicator = document.getElementById('api-status');
    if (indicator) {
        indicator.className = `api-status ${isOnline ? 'online' : 'offline'}`;
        indicator.textContent = isOnline ? 'API \u5DF2\u8FDE\u63A5' : 'API \u672A\u8FDE\u63A5';
    }
}

/**
 * Extract the bin label from a raw WOE-table record.
 * The key is `_bin_{feature}` but varies per feature.
 * @param {object} record - WOE table record
 * @param {string} feature - Feature name
 * @returns {string}
 */
function extractBinLabel(record, feature) {
    const binKey = `_bin_${feature}`;
    if (record[binKey] !== undefined) return String(record[binKey]);
    for (const key in record) {
        if (key.startsWith('_bin_')) return String(record[key]);
    }
    return record.bin || record.bin_label || 'N/A';
}

/**
 * Render a generic data table from an array of records.
 * @param {Array<object>} records - Array of row objects
 * @param {string} containerId - Target element ID
 * @param {number} maxRows - Max rows to display
 */
function renderDataTable(records, containerId, maxRows = 20) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!records || records.length === 0) {
        container.innerHTML = '<p class="text-muted">\u6682\u65E0\u6570\u636E</p>';
        return;
    }
    const cols = Object.keys(records[0]);
    let html = '<table class="data-table"><thead><tr>';
    cols.forEach(c => { html += `<th>${escapeHTML(c)}</th>`; });
    html += '</tr></thead><tbody>';
    records.slice(0, maxRows).forEach(row => {
        html += '<tr>';
        cols.forEach(c => { html += `<td>${escapeHTML(String(row[c] ?? ''))}</td>`; });
        html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

/* ============================================================
 * Screen Navigation
 * ============================================================ */

/**
 * Switch to a named screen: hide all, show target, update nav.
 * Also renders any previously-loaded data for the target screen.
 * @param {string} name - Screen name (must be in SCREENS)
 */
function switchScreen(name) {
    if (!SCREENS.includes(name)) {
        console.warn(`Unknown screen: ${name}`);
        return;
    }
    // Hide all screens
    SCREENS.forEach(s => {
        const el = document.getElementById(`screen-${s}`);
        if (el) el.classList.remove('active');
    });
    // Show target
    const target = document.getElementById(`screen-${name}`);
    if (target) target.classList.add('active');
    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.screen === name);
    });
    state.currentScreen = name;
    // Render existing data if available
    _renderExistingData(name);
}

/**
 * Render previously-loaded data when a screen becomes visible.
 * @param {string} screen
 */
function _renderExistingData(screen) {
    switch (screen) {
        case 'eda':
            if (state.edaResult) _renderAllEDA(state.edaResult);
            break;
        case 'binning':
            if (state.ivRanking) _renderIVRankingContent(state.ivRanking);
            if (state.selectedFeature) renderWOETable(state.selectedFeature);
            break;
        case 'training':
            if (state.trainResult) renderModelResult(state.trainResult);
            if (state.scorecard) renderScorecard(state.scorecard);
            break;
        case 'evaluation':
            if (state.evalResult) renderEvaluation(state.evalResult);
            break;
    }
}

/* ============================================================
 * Screen: Data Upload
 * ============================================================ */

/**
 * Load the built-in sample dataset.
 * Calls POST /api/v1/data/load-sample
 */
async function loadSample() {
    try {
        const res = await api.post('/api/v1/data/load-sample');
        if (res.status === 'ok') {
            state.dataSummary = res.summary;
            state.dataPreview = res.preview;
            displaySummary(res);
            showToast('\u6837\u4F8B\u6570\u636E\u52A0\u8F7D\u6210\u529F', 'success');
        } else {
            showToast('\u52A0\u8F7D\u5931\u8D25', 'error');
        }
    } catch (error) {
        showToast(`\u52A0\u8F7D\u5931\u8D25: ${error.message}`, 'error');
    }
}

/**
 * Handle file upload from the file input.
 * Calls POST /api/v1/data/upload with FormData.
 */
async function uploadFile() {
    const input = document.getElementById('file-input');
    if (!input || !input.files || input.files.length === 0) {
        showToast('\u8BF7\u5148\u9009\u62E9\u6587\u4EF6', 'warning');
        return;
    }
    const file = input.files[0];
    var ts = document.getElementById('target-col-select');
    const targetCol = (ts && ts.value) ? ts.value : 'class';
    try {
        const text = await file.text();
        let res;
        if (CONFIG.apiBase === 'local' && typeof window.RiskEngineLocal !== 'undefined') {
            res = await api.post('/api/v1/data/upload', { text: text, target_col: targetCol });
        } else {
            const formData = new FormData();
            formData.append('file', file);
            if (targetCol) formData.append('target_col', targetCol);
            res = await api.upload('/api/v1/data/upload', formData);
        }
        if (res.status === 'ok') {
            state.dataSummary = res.summary;
            state.dataPreview = res.preview;
            displaySummary(res);
            showToast(`\u6587\u4EF6 "${escapeHTML(file.name)}" \u4E0A\u4F20\u6210\u529F`, 'success');
        } else {
            showToast('\u4E0A\u4F20\u5931\u8D25', 'error');
        }
    } catch (error) {
        showToast(`\u4E0A\u4F20\u5931\u8D25: ${error.message}`, 'error');
    }
}

/**
 * Display the data quality summary and preview table.
 * @param {object} data - API response { status, summary, preview }
 */
function displaySummary(data) {
    const summary = data.summary || data;
    const preview = data.preview || [];
    const container = document.getElementById('data-summary-content');
    if (!container) return;

    const total = summary.total_samples || 0;
    const numFeatures = summary.num_features || 0;
    const goodCount = summary.good_count || 0;
    const badCount = summary.bad_count || 0;
    const goodRate = summary.good_rate || 0;
    const badRate = summary.bad_rate || 0;
    const numCols = summary.numeric_cols || [];
    const catCols = summary.categorical_cols || [];
    const missing = summary.missing_values || {};

    // Stat cards
    let html = '<div class="stat-grid">';
    html += _statCard('\u603B\u6837\u672C\u6570', formatNumber(total, 0), '#3b82f6');
    html += _statCard('\u7279\u5F81\u6570', formatNumber(numFeatures, 0), '#8b5cf6');
    html += _statCard('\u597D\u5BA2\u6237', `${formatNumber(goodCount, 0)} (${formatPercent(goodRate)})`, '#10b981');
    html += _statCard('\u574F\u5BA2\u6237', `${formatNumber(badCount, 0)} (${formatPercent(badRate)})`, '#ef4444');
    html += _statCard('\u6570\u503C\u578B\u5B57\u6BB5', formatNumber(numCols.length, 0), '#06b6d4');
    html += _statCard('\u7C7B\u522B\u578B\u5B57\u6BB5', formatNumber(catCols.length, 0), '#f59e0b');
    html += _statCard('\u76EE\u6807\u5217', escapeHTML(summary.target_col || 'class'), '#6366f1');
    html += _statCard('\u7F3A\u5931\u5B57\u6BB5\u6570', formatNumber(Object.keys(missing).length, 0), '#ec4899');
    html += '</div>';

    // Missing values detail
    if (Object.keys(missing).length > 0) {
        html += '<h4>\u7F3A\u5931\u503C\u660E\u7EC6</h4><table class="data-table"><thead><tr><th>\u5B57\u6BB5</th><th>\u7F3A\u5931\u6570</th><th>\u7F3A\u5931\u7387</th></tr></thead><tbody>';
        for (const [col, count] of Object.entries(missing)) {
            const rate = total > 0 ? count / total : 0;
            html += `<tr><td>${escapeHTML(col)}</td><td>${formatNumber(count, 0)}</td><td>${formatPercent(rate)}</td></tr>`;
        }
        html += '</tbody></table>';
    }

    container.innerHTML = html;

    // Preview table
    renderDataTable(preview, 'data-preview-content', 5);

    // Show the cards
    const summaryCard = document.getElementById('data-summary-card');
    if (summaryCard) summaryCard.style.display = 'block';
    const previewCard = document.getElementById('data-preview-card');
    if (previewCard) previewCard.style.display = 'block';

    // Populate target column dropdown
    var tc = document.getElementById('target-config-section');
    var ts = document.getElementById('target-col-select');
    if (ts && preview && preview.length > 0) {
        var cols = Object.keys(preview[0]);
        var ct = summary.target_col || summary.target || 'class';
        var opts = '<option value="">-- \u8BF7\u9009\u62E9 --</option>';
        cols.forEach(function(col) {
            var sel = col === ct ? ' selected' : '';
            opts += '<option value="' + escapeHTML(col) + '"' + sel + '>' + escapeHTML(col) + '</option>';
        });
        ts.innerHTML = opts;
        state.targetCol = ct;
        if (tc) tc.style.display = 'block';
    }
    renderFieldTypes(summary, preview);
}

/**
 * Helper: generate a stat card HTML string.
 * @param {string} label
 * @param {string} value
 * @param {string} color
 * @returns {string}
 */
function _statCard(label, value, color) {
    return `<div class="stat-card" style="border-left-color:${color};">`
        + `<div class="stat-label">${escapeHTML(label)}</div>`
        + `<div class="stat-value" style="color:${color};">${value}</div>`
        + `</div>`;
}

/* ============================================================
 * Screen: EDA (Exploratory Data Analysis)
 * ============================================================ */

/**
 * Run EDA analysis.
 * Calls POST /api/v1/eda/run
 */
async function runEDA() {
    try {
        const res = await api.post('/api/v1/eda/run');
        if (res.status === 'ok') {
            state.edaResult = res.result;
            _renderAllEDA(res.result);
            showToast('EDA \u5206\u6790\u5B8C\u6210', 'success');
        } else {
            showToast('EDA \u5206\u6790\u5931\u8D25', 'error');
        }
    } catch (error) {
        showToast(`EDA \u5206\u6790\u5931\u8D25: ${error.message}`, 'error');
    }
}

/**
 * Render all EDA results (quality, numeric, categorical, correlation, target, missing).
 * @param {object} result - EDA result object
 */
function _renderAllEDA(result) {
    // Data quality
    const dq = result.data_quality;
    if (dq) {
        let html = '<div class="stat-grid">';
        html += _statCard('\u603B\u884C\u6570', formatNumber(dq.total_rows, 0), '#3b82f6');
        html += _statCard('\u603B\u5217\u6570', formatNumber(dq.total_cols, 0), '#8b5cf6');
        html += _statCard('\u7F3A\u5931\u5355\u5143\u683C', formatNumber(dq.missing_cells, 0), '#ef4444');
        html += _statCard('\u7F3A\u5931\u7387', formatPercent(dq.missing_rate), '#f59e0b');
        html += _statCard('\u91CD\u590D\u884C', formatNumber(dq.duplicate_rows, 0), '#ec4899');
        html += _statCard('\u5185\u5B58 (MB)', formatNumber(dq.memory_mb, 2), '#06b6d4');
        html += '</div>';
        setContainerHTML('eda-quality-stats', html);
    }

    if (result.numeric_stats) renderNumericStats(result.numeric_stats);
    if (result.categorical_stats) renderCategoricalStats(result.categorical_stats);
    if (result.correlation) renderCorrelation(result.correlation);
    if (result.target_distribution) renderTargetDist(result.target_distribution);

    // Missing report
    if (result.missing_report && result.missing_report.length > 0) {
        let html = '<table class="data-table"><thead><tr><th>\u5B57\u6BB5</th><th>\u7F3A\u5931\u6570</th><th>\u7F3A\u5931\u7387</th></tr></thead><tbody>';
        result.missing_report.forEach(r => {
            html += `<tr><td>${escapeHTML(r.column)}</td><td>${formatNumber(r.missing_count, 0)}</td><td>${formatPercent(r.missing_rate)}</td></tr>`;
        });
        html += '</tbody></table>';
        setContainerHTML('eda-missing-report', html);
    } else {
        setContainerHTML('eda-missing-report', '<p class="text-muted">\u65E0\u7F3A\u5931\u503C</p>');
    }
    document.getElementById('eda-content').style.display = 'block';
}

function renderFieldTypes(summary, preview) {
    var container = document.getElementById('data-fieldtypes-content');
    var card = document.getElementById('data-fieldtypes-card');
    if (!container || !preview || preview.length === 0) return;
    var cols = Object.keys(preview[0]);
    var types = summary.column_types || summary.dtypes || {};
    var rows = '';
    cols.forEach(function(col) {
        var t = types[col] || 'unknown';
        var tL = String(t).toLowerCase();
        var bc = 'field-type-categorical';
        if (tL.indexOf('int') >= 0 || tL.indexOf('float') >= 0 || tL.indexOf('number') >= 0) bc = 'field-type-numeric';
        else if (tL.indexOf('bool') >= 0 || tL.indexOf('binary') >= 0) bc = 'field-type-binary';
        var s = preview[0][col];
        var ss = s !== null && s !== undefined ? String(s).substring(0, 30) : '--';
        rows += '<tr><td style="font-weight:600">' + escapeHTML(col) + '</td><td><span class="field-type-badge ' + bc + '">' + escapeHTML(String(t)) + '</span></td><td>' + escapeHTML(ss) + '</td></tr>';
    });
    container.innerHTML = '<table class="data-table"><thead><tr><th>\u5B57\u6BB5\u540D</th><th>\u7C7B\u578B</th><th>\u793A\u4F8B\u503C</th></tr></thead><tbody>' + rows + '</tbody></table>';
    if (card) card.style.display = 'block';
}

function renderFieldTypes(summary, preview) {
    var container = document.getElementById('data-fieldtypes-content');
    var card = document.getElementById('data-fieldtypes-card');
    if (!container || !preview || preview.length === 0) return;
    var cols = Object.keys(preview[0]);
    var types = summary.column_types || summary.dtypes || {};
    var rows = '';
    cols.forEach(function(col) {
        var t = types[col] || 'unknown';
        var tL = String(t).toLowerCase();
        var bc = 'field-type-categorical';
        if (tL.indexOf('int') >= 0 || tL.indexOf('float') >= 0 || tL.indexOf('number') >= 0) bc = 'field-type-numeric';
        else if (tL.indexOf('bool') >= 0 || tL.indexOf('binary') >= 0) bc = 'field-type-binary';
        var s = preview[0][col];
        var ss = s !== null && s !== undefined ? String(s).substring(0, 30) : '--';
        rows += '<tr><td style="font-weight:600">' + escapeHTML(col) + '</td><td><span class="field-type-badge ' + bc + '">' + escapeHTML(String(t)) + '</span></td><td>' + escapeHTML(ss) + '</td></tr>';
    });
    container.innerHTML = '<table class="data-table"><thead><tr><th>\u5B57\u6BB5\u540D</th><th>\u7C7B\u578B</th><th>\u793A\u4F8B\u503C</th></tr></thead><tbody>' + rows + '</tbody></table>';
    if (card) card.style.display = 'block';
}

/**
 * Render numeric feature statistics table.
 * @param {Array<object>} stats - Array of { column, mean, std, min, max, ... }
 */
function renderNumericStats(stats) {
    if (!stats || stats.length === 0) {
        setContainerHTML('eda-numeric-content', '<p class="text-muted">\u65E0\u6570\u503C\u578B\u5B57\u6BB5</p>');
        return;
    }
    let html = '<table class="data-table"><thead><tr>'
        + '<th>\u5B57\u6BB5</th><th>\u5747\u503C</th><th>\u6807\u51C6\u5DEE</th><th>\u6700\u5C0F\u503C</th>'
        + '<th>\u4E2D\u4F4D\u6570</th><th>\u6700\u5927\u503C</th><th>\u504F\u5EA6</th><th>\u7F3A\u5931</th>'
        + '</tr></thead><tbody>';
    stats.forEach(s => {
        html += `<tr>`
            + `<td>${escapeHTML(s.column)}</td>`
            + `<td>${formatNumber(s.mean)}</td>`
            + `<td>${formatNumber(s.std)}</td>`
            + `<td>${formatNumber(s.min)}</td>`
            + `<td>${formatNumber(s.median)}</td>`
            + `<td>${formatNumber(s.max)}</td>`
            + `<td>${formatNumber(s.skew, 4)}</td>`
            + `<td>${formatNumber(s.missing, 0)}</td>`
            + `</tr>`;
    });
    html += '</tbody></table>';
    setContainerHTML('eda-numeric-content', html);
}

/**
 * Render categorical feature statistics table.
 * @param {Array<object>} stats - Array of { column, unique_count, top_values, missing }
 */
function renderCategoricalStats(stats) {
    if (!stats || stats.length === 0) {
        setContainerHTML('eda-categorical-content', '<p class="text-muted">\u65E0\u7C7B\u522B\u578B\u5B57\u6BB5</p>');
        return;
    }
    let html = '<table class="data-table"><thead><tr>'
        + '<th>\u5B57\u6BB5</th><th>\u552F\u4E00\u503C\u6570</th><th>Top \u7C7B\u522B</th><th>\u9891\u6B21</th><th>\u7F3A\u5931</th>'
        + '</tr></thead><tbody>';
    stats.forEach(s => {
        const topEntries = Object.entries(s.top_values || {});
        const topCat = topEntries.length > 0 ? escapeHTML(topEntries[0][0]) : '-';
        const topFreq = topEntries.length > 0 ? formatNumber(topEntries[0][1], 0) : '-';
        html += `<tr>`
            + `<td>${escapeHTML(s.column)}</td>`
            + `<td>${formatNumber(s.unique_count, 0)}</td>`
            + `<td>${topCat}</td>`
            + `<td>${topFreq}</td>`
            + `<td>${formatNumber(s.missing, 0)}</td>`
            + `</tr>`;
    });
    html += '</tbody></table>';
    setContainerHTML('eda-categorical-content', html);
}

/**
 * Render a correlation heatmap using colored cells.
 * @param {object} data - { columns: [...], matrix: [[...], ...] }
 */
function renderCorrelation(data) {
    const container = document.getElementById('eda-correlation-content');
    if (!container) return;
    const cols = data.columns || [];
    const matrix = data.matrix || [];
    if (cols.length === 0 || matrix.length === 0) {
        container.innerHTML = '<p class="text-muted">\u65E0\u76F8\u5173\u6027\u6570\u636E</p>';
        return;
    }

    let html = '<div class="heatmap-wrapper"><table class="heatmap"><thead><tr><th></th>';
    // Truncate long column names for display
    const shortCols = cols.map(c => c.length > 10 ? c.substring(0, 8) + '..' : c);
    shortCols.forEach(c => { html += `<th title="${escapeHTML(c)}">${escapeHTML(c)}</th>`; });
    html += '</tr></thead><tbody>';
    for (let i = 0; i < matrix.length; i++) {
        html += `<tr><th title="${escapeHTML(cols[i])}">${escapeHTML(shortCols[i])}</th>`;
        for (let j = 0; j < matrix[i].length; j++) {
            const val = matrix[i][j];
            const bg = correlationColor(val);
            const txt = Math.abs(val) > 0.5 ? '#fff' : '#1e293b';
            html += `<td class="heatmap-cell" style="background:${bg};color:${txt};" title="${escapeHTML(cols[i])} vs ${escapeHTML(cols[j])} = ${val}">${formatNumber(val, 2)}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table></div>';
    container.innerHTML = html;
}

/**
 * Render target variable distribution as a bar chart using div elements.
 * @param {object} data - { target_col, counts: {label: count}, rates: {...}, imbalance_ratio }
 */
function renderTargetDist(data) {
    const container = document.getElementById('eda-target-content');
    if (!container) return;
    const counts = data.counts || {};
    const rates = data.rates || {};
    const entries = Object.entries(counts);
    if (entries.length === 0) {
        container.innerHTML = '<p class="text-muted">\u65E0\u76EE\u6807\u5206\u5E03\u6570\u636E</p>';
        return;
    }
    const maxCount = Math.max(...entries.map(e => e[1]));
    const colors = ['#10b981', '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6'];

    let html = `<p class="text-muted">\u76EE\u6807\u5217: ${escapeHTML(data.target_col || '')} | \u4E0D\u5E73\u8861\u6BD4: ${formatNumber(data.imbalance_ratio, 2)}</p>`;
    html += '<div class="bar-chart">';
    entries.forEach(([label, count], i) => {
        const rate = rates[label] || 0;
        const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
        const color = colors[i % colors.length];
        html += `<div class="bar-chart-col">`
            + `<div class="bar-chart-bar" style="height:${height}%;background-color:${color};" title="${escapeHTML(label)}: ${count}">`
            + `<span class="bar-chart-count">${formatNumber(count, 0)}</span>`
            + `</div>`
            + `<div class="bar-chart-label">${escapeHTML(label)} (${formatPercent(rate)})</div>`
            + `</div>`;
    });
    html += '</div>';
    container.innerHTML = html;
}

/* ============================================================
 * Screen: Binning (WOE / IV)
 * ============================================================ */

/**
 * Run WOE/IV binning.
 * Calls POST /api/v1/binning/run
 */
async function runBinning() {
    try {
        var mb = document.getElementById('max-bins');
        var it = document.getElementById('iv-threshold');
        var maxBins = mb ? parseInt(mb.value) : 5;
        var ivThresh = it ? parseFloat(it.value) : 0.02;
        const res = await api.post('/api/v1/binning/run?max_bins=' + maxBins + '&iv_threshold=' + ivThresh);
        if (res.status === 'ok') {
            state.binningResult = res.result;
            state.selectedFeatures = res.selected_features;
            // Also fetch IV ranking
            await renderIVRanking();
            _renderBinningSummary(res.result);
            showToast('\u5206\u7BB1\u5B8C\u6210', 'success');
        } else {
            showToast('\u5206\u7BB1\u5931\u8D25', 'error');
        }
    } catch (error) {
        showToast(`\u5206\u7BB1\u5931\u8D25: ${error.message}`, 'error');
    }
}

/**
 * Render binning summary stats.
 * @param {object} result - Binning result
 */
function _renderBinningSummary(result) {
    let html = '<div class="stat-grid">';
    html += _statCard('\u603B\u7279\u5F81\u6570', formatNumber(result.total_features, 0), '#3b82f6');
    html += _statCard('\u5F3A\u9884\u6D4B\u529B', formatNumber(result.strong_features, 0), '#10b981');
    html += _statCard('\u4E2D\u9884\u6D4B\u529B', formatNumber(result.medium_features, 0), '#f59e0b');
    html += _statCard('\u5F31\u9884\u6D4B\u529B', formatNumber(result.weak_features, 0), '#6b7280');
    html += '</div>';
    setContainerHTML('binning-stats', html);
    var binningContent = document.getElementById('binning-content');
    if (binningContent) binningContent.style.display = 'block';
    if (result.features && result.features.length > 0) {
        var ff = result.features[0].feature;
        if (ff) { try { renderWOETable(ff); } catch(e) {} }
    }
}

/**
 * Fetch and render IV ranking.
 * Calls GET /api/v1/binning/iv-ranking
 */
async function renderIVRanking() {
    try {
        const res = await api.get('/api/v1/binning/iv-ranking');
        if (res.ranking) {
            state.ivRanking = res.ranking;
            _renderIVRankingContent(res.ranking);
        }
    } catch (error) {
        // Try to use state from binning result
        if (state.binningResult && state.binningResult.features) {
            const ranking = state.binningResult.features.map(f => ({
                feature: f.feature, iv: f.iv, strength: f.strength,
            }));
            state.ivRanking = ranking;
            _renderIVRankingContent(ranking);
        } else {
            showToast(`\u52A0\u8F7D IV \u6392\u540D\u5931\u8D25: ${error.message}`, 'error');
        }
    }
}

/**
 * Render IV ranking content (sorted feature list with IV bars).
 * @param {Array<object>} ranking - [{ feature, iv, strength }]
 */
function _renderIVRankingContent(ranking) {
    const container = document.getElementById('iv-ranking-content');
    if (!container) return;
    if (!ranking || ranking.length === 0) {
        container.innerHTML = '<p class="text-muted">\u8BF7\u5148\u6267\u884C\u5206\u7BB1</p>';
        return;
    }
    const maxIV = Math.max(...ranking.map(r => r.iv || 0), 0.5);
    let html = '<div class="iv-ranking-list">';
    ranking.forEach(item => {
        const color = colorIV(item.iv);
        const active = state.selectedFeature === item.feature ? ' feature-active' : '';
        html += `<div class="feature-item${active}" onclick="selectFeature('${escapeHTML(item.feature)}')">`
            + `<div class="feature-name">${escapeHTML(item.feature)}</div>`
            + `<div class="feature-iv-bar">${renderBar(item.iv, maxIV, color)}</div>`
            + `<span class="feature-strength" style="color:${color};">${escapeHTML(item.strength || '')}</span>`
            + `</div>`;
    });
    html += '</div>';
    container.innerHTML = html;
}

/**
 * Select a feature and render its WOE table.
 * Calls GET /api/v1/binning/woe-table/{feature}
 * @param {string} feature - Feature name
 */
async function renderWOETable(feature) {
    selectFeature(feature);
    try {
        const res = await api.get(`/api/v1/binning/woe-table/${encodeURIComponent(feature)}`);
        if (res.woe_table) {
            _renderWOETableContent(feature, res.woe_table);
        }
    } catch (error) {
        // Fallback: use binning result bins
        if (state.binningResult) {
            const featData = state.binningResult.features.find(f => f.feature === feature);
            if (featData && featData.bins) {
                _renderWOEBinsContent(feature, featData.bins);
                return;
            }
        }
        setContainerHTML('woe-detail-content', `<p class="text-muted">\u52A0\u8F7D WOE \u8868\u5931\u8D25: ${escapeHTML(error.message)}</p>`);
    }
}

/**
 * Render WOE table from raw API records (from /woe-table endpoint).
 * @param {string} feature
 * @param {Array<object>} records - Raw DataFrame records
 */
function _renderWOETableContent(feature, records) {
    const container = document.getElementById('woe-detail-content');
    if (!container) return;
    if (!records || records.length === 0) {
        container.innerHTML = '<p class="text-muted">\u65E0 WOE \u6570\u636E</p>';
        return;
    }
    let html = `<h4>WOE \u8868 - ${escapeHTML(feature)}</h4>`;
    html += '<table class="data-table"><thead><tr>'
        + '<th>\u5206\u7BB1</th><th>\u603B\u6570</th><th>\u574F\u5BA2\u6237</th><th>\u597D\u5BA2\u6237</th>'
        + '<th>\u574F\u5BA2\u6237\u7387</th><th>WOE</th><th>IV</th><th>\u5360\u6BD4</th>'
        + '</tr></thead><tbody>';
    records.forEach(r => {
        const binLabel = extractBinLabel(r, feature);
        const woe = r.woe !== undefined ? Number(r.woe) : null;
        const woeColor = woe !== null ? colorWOE(woe) : '#6b7280';
        const total = r.total || 0;
        const bad = r.bad !== undefined ? r.bad : 0;
        const good = r.good !== undefined ? r.good : 0;
        const badRate = total > 0 && bad !== undefined ? bad / total : 0;
        const pct = r.total_pct !== undefined ? r.total_pct : (r.pct || 0);
        html += `<tr>`
            + `<td>${escapeHTML(binLabel)}</td>`
            + `<td>${formatNumber(total, 0)}</td>`
            + `<td>${formatNumber(bad, 0)}</td>`
            + `<td>${formatNumber(good, 0)}</td>`
            + `<td>${formatPercent(badRate)}</td>`
            + `<td style="color:${woeColor};font-weight:bold;">${formatNumber(woe, 4)}</td>`
            + `<td>${formatNumber(r.iv, 4)}</td>`
            + `<td>${formatPercent(pct)}</td>`
            + `</tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

/**
 * Render WOE table from binning result bins (fallback).
 * @param {string} feature
 * @param {Array<object>} bins - [{ bin, total, bad, good, woe, iv, pct, bad_rate }]
 */
function _renderWOEBinsContent(feature, bins) {
    const container = document.getElementById('woe-detail-content');
    if (!container) return;
    let html = `<h4>WOE \u8868 - ${escapeHTML(feature)}</h4>`;
    html += '<table class="data-table"><thead><tr>'
        + '<th>\u5206\u7BB1</th><th>\u603B\u6570</th><th>\u574F\u5BA2\u6237</th><th>\u597D\u5BA2\u6237</th>'
        + '<th>\u574F\u5BA2\u6237\u7387</th><th>WOE</th><th>IV</th><th>\u5360\u6BD4</th>'
        + '</tr></thead><tbody>';
    bins.forEach(b => {
        const woe = b.woe;
        const woeColor = colorWOE(woe);
        html += `<tr>`
            + `<td>${escapeHTML(b.bin)}</td>`
            + `<td>${formatNumber(b.total, 0)}</td>`
            + `<td>${formatNumber(b.bad, 0)}</td>`
            + `<td>${formatNumber(b.good, 0)}</td>`
            + `<td>${formatPercent(b.bad_rate)}</td>`
            + `<td style="color:${woeColor};font-weight:bold;">${formatNumber(woe, 4)}</td>`
            + `<td>${formatNumber(b.iv, 4)}</td>`
            + `<td>${formatPercent(b.pct)}</td>`
            + `</tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

/**
 * Highlight the selected feature in the IV ranking list.
 * @param {string} feature - Feature name
 */
function selectFeature(feature) {
    state.selectedFeature = feature;
    // Update visual selection
    document.querySelectorAll('.feature-item').forEach(el => {
        const nameEl = el.querySelector('.feature-name');
        el.classList.toggle('feature-active', nameEl && nameEl.textContent === feature);
    });
}

/* ============================================================
 * Screen: Training
 * ============================================================ */

/**
 * Run model training.
 * Calls POST /api/v1/training/run
 */
async function runTraining() {
    try {
        var splitSlider = document.getElementById('split-ratio-slider');
        var testSize = 0.3;
        if (splitSlider) { var trainPct = parseInt(splitSlider.value); testSize = (100 - trainPct) / 100; }
        var ivThresh = document.getElementById('iv-threshold');
        var ivVal = ivThresh ? parseFloat(ivThresh.value) : 0.02;
        const res = await api.post('/api/v1/training/run?test_size=' + testSize + '&iv_threshold=' + ivVal);
        if (res.status === 'ok') {
            state.trainResult = res.result;
            state.evalResult = res.evaluation;
            renderModelResult(res.result);
            // Fetch full scorecard
            try {
                const scRes = await api.get('/api/v1/scorecard/table');
                if (scRes.scorecard) {
                    state.scorecard = scRes.scorecard;
                    renderScorecard(scRes.scorecard);
                }
            } catch (scErr) {
                // Use preview from training result
                if (res.result.scorecard_preview) {
                    state.scorecard = res.result.scorecard_preview;
                    renderScorecard(res.result.scorecard_preview);
                }
            }
            showToast('\u6A21\u578B\u8BAD\u7EC3\u5B8C\u6210', 'success');
        } else {
            showToast('\u8BAD\u7EC3\u5931\u8D25', 'error');
        }
    } catch (error) {
        showToast(`\u8BAD\u7EC3\u5931\u8D25: ${error.message}`, 'error');
    }
}

/**
 * Render model training result (coefficients, intercept, VIF, metrics).
 * @param {object} data - Training result
 */
function renderModelResult(data) {
    const container = document.getElementById('model-coef-content');
    if (!container) return;

    // Model info
    let html = '<div class="stat-grid">';
    html += _statCard('\u6A21\u578B\u7C7B\u578B', escapeHTML(data.model_type || 'Logistic Regression'), '#3b82f6');
    html += _statCard('\u7279\u5F81\u6570', formatNumber(data.n_features, 0), '#8b5cf6');
    html += _statCard('\u57FA\u7840\u5206', formatNumber(data.base_score, 0), '#06b6d4');
    html += _statCard('PDO', formatNumber(data.pdo, 0), '#f59e0b');
    html += _statCard('\u622A\u8DDD', formatNumber(data.intercept, 6), '#ec4899');
    html += _statCard('\u603B\u5206\u7BB1\u6570', formatNumber(data.scorecard_total_bins, 0), '#10b981');
    html += '</div>';

    // Train / Test metrics
    const tm = data.train_metrics || {};
    const em = data.test_metrics || {};
    html += '<h4>\u8BAD\u7EC3 / \u6D4B\u8BD5\u6307\u6807</h4>';
    html += '<table class="data-table"><thead><tr><th>\u6307\u6807</th><th>\u8BAD\u7EC3\u96C6</th><th>\u6D4B\u8BD5\u96C6</th></tr></thead><tbody>';
    html += `<tr><td>AUC</td><td>${formatNumber(tm.auc, 4)}</td><td>${formatNumber(em.auc, 4)}</td></tr>`;
    html += `<tr><td>KS</td><td>${formatNumber(tm.ks, 4)}</td><td>${formatNumber(em.ks, 4)}</td></tr>`;
    html += `<tr><td>Gini</td><td>${formatNumber(tm.gini, 4)}</td><td>${formatNumber(em.gini, 4)}</td></tr>`;
    html += `<tr><td>\u6837\u672C\u6570</td><td>${formatNumber(tm.n_samples, 0)}</td><td>${formatNumber(em.n_samples, 0)}</td></tr>`;
    html += '</tbody></table>';

    // Coefficients
    const coef = data.coef || {};
    html += '<h4>\u6A21\u578B\u7CFB\u6570</h4>';
    html += '<table class="data-table"><thead><tr><th>\u7279\u5F81 (WOE)</th><th>\u7CFB\u6570</th></tr></thead><tbody>';
    for (const [k, v] of Object.entries(coef)) {
        const sign = v >= 0 ? '+' : '';
        const color = v >= 0 ? '#10b981' : '#ef4444';
        html += `<tr><td>${escapeHTML(k)}</td><td style="color:${color};font-weight:bold;">${sign}${formatNumber(v, 6)}</td></tr>`;
    }
    html += `<tr><td><strong>\u622A\u8DDD</strong></td><td><strong>${formatNumber(data.intercept, 6)}</strong></td></tr>`;
    html += '</tbody></table>';

    // VIF
    const vif = data.vif || {};
    if (Object.keys(vif).length > 0) {
        html += '<h4>VIF (\u65B9\u5DEE\u81A8\u80C0\u56E0\u5B50)</h4>';
        html += '<table class="data-table"><thead><tr><th>\u7279\u5F81</th><th>VIF</th><th>\u72B6\u6001</th></tr></thead><tbody>';
        for (const [k, v] of Object.entries(vif)) {
            const isInf = v === Infinity || v === 'inf' || !isFinite(v);
            const display = isInf ? 'inf' : formatNumber(v, 2);
            const ok = !isInf && v < 10;
            const status = ok ? '<span style="color:#10b981">\u6B63\u5E38</span>' : '<span style="color:#ef4444">\u8B66\u544A</span>';
            html += `<tr><td>${escapeHTML(k)}</td><td>${display}</td><td>${status}</td></tr>`;
        }
        html += '</tbody></table>';
    }

    // Selected features
    if (data.features && data.features.length > 0) {
        html += '<h4>\u5165\u9009\u7279\u5F81</h4><div class="tag-list">';
        data.features.forEach(f => {
            html += `<span class="tag">${escapeHTML(f)}</span>`;
        });
        html += '</div>';
    }

    container.innerHTML = html;
    var trainingContent = document.getElementById('training-content');
    if (trainingContent) trainingContent.style.display = 'block';
}

/**
 * Render the scorecard table with color coding.
 * @param {Array<object>} data - Scorecard array [{ feature, bin, woe, coef, score }]
 */
function renderScorecard(data) {
    const container = document.getElementById('scorecard-content');
    if (!container) return;
    if (!data || data.length === 0) {
        container.innerHTML = '<p class="text-muted">\u8BF7\u5148\u8BAD\u7EC3\u6A21\u578B</p>';
        return;
    }

    // Group by feature for better readability
    const grouped = {};
    data.forEach(item => {
        if (!grouped[item.feature]) grouped[item.feature] = [];
        grouped[item.feature].push(item);
    });

    let html = '<table class="data-table scorecard-table"><thead><tr>'
        + '<th>\u7279\u5F81</th><th>\u5206\u7BB1</th><th>WOE</th><th>\u7CFB\u6570</th><th>\u52A0\u51CF\u5206</th>'
        + '</tr></thead><tbody>';
    for (const [feature, bins] of Object.entries(grouped)) {
        bins.forEach((b, idx) => {
            const score = b.score;
            const scoreColor = score >= 0 ? '#10b981' : '#ef4444';
            const sign = score >= 0 ? '+' : '';
            const woeColor = colorWOE(b.woe);
            html += `<tr>`
                + `<td>${idx === 0 ? escapeHTML(feature) : ''}</td>`
                + `<td>${escapeHTML(b.bin)}</td>`
                + `<td style="color:${woeColor};">${formatNumber(b.woe, 4)}</td>`
                + `<td>${formatNumber(b.coef, 6)}</td>`
                + `<td style="color:${scoreColor};font-weight:bold;">${sign}${formatNumber(score, 1)}</td>`
                + `</tr>`;
        });
    }
    html += '</tbody></table>';
    container.innerHTML = html;
}

/* ============================================================
 * Screen: Evaluation
 * ============================================================ */

/**
 * Render the full evaluation screen (metrics, KS curve, lift, score dist).
 * @param {object} data - Evaluation result { auc, ks, gini, lift, ks_curve, score_distribution, confusion }
 */
function renderEvaluation(data) {
    if (!data) {
        setContainerHTML('eval-metrics', '<p class="text-muted">\u8BF7\u5148\u8BAD\u7EC3\u6A21\u578B</p>');
        return;
    }

    // Metric cards
    let html = '<div class="stat-grid">';
    html += _statCard('KS', formatNumber(data.ks, 4), '#3b82f6');
    html += _statCard('AUC', formatNumber(data.auc, 4), '#10b981');
    html += _statCard('Gini', formatNumber(data.gini, 4), '#8b5cf6');
    html += _statCard('\u6837\u672C\u6570', formatNumber(data.n_samples, 0), '#06b6d4');
    html += '</div>';

    // Confusion matrix
    const cm = data.confusion;
    if (cm) {
        html += '<h4>\u6DF7\u6DC6\u77E9\u9635 (threshold=' + formatNumber(cm.threshold, 2) + ')</h4>';
        html += '<div class="stat-grid">';
        html += _statCard('TP', formatNumber(cm.tp, 0), '#10b981');
        html += _statCard('FP', formatNumber(cm.fp, 0), '#f59e0b');
        html += _statCard('TN', formatNumber(cm.tn, 0), '#3b82f6');
        html += _statCard('FN', formatNumber(cm.fn, 0), '#ef4444');
        html += _statCard('Precision', formatNumber(cm.precision, 4), '#8b5cf6');
        html += _statCard('Recall', formatNumber(cm.recall, 4), '#06b6d4');
        html += _statCard('F1', formatNumber(cm.f1, 4), '#ec4899');
        html += '</div>';
    }
    setContainerHTML('eval-metrics', html);

    // Charts
    if (data.ks_curve) renderKSCurve(data.ks_curve);
    if (data.lift) renderLiftTable(data.lift);
    if (data.score_distribution) renderScoreDist(data.score_distribution);
    var evalContent = document.getElementById('eval-content');
    if (evalContent) evalContent.style.display = 'block';
}

/**
 * Render KS curve as an SVG line chart.
 * @param {object} data - { ks_max, ks_threshold, fpr: [...], tpr: [...], thresholds: [...] }
 */
function renderKSCurve(data) {
    const container = document.getElementById('ks-curve-content');
    if (!container) return;
    const fpr = data.fpr || [];
    const tpr = data.tpr || [];
    if (fpr.length === 0 || tpr.length === 0) {
        container.innerHTML = '<p class="text-muted">\u65E0 KS \u66F2\u7EBF\u6570\u636E</p>';
        return;
    }

    const W = 600, H = 400, pad = { top: 30, right: 30, bottom: 50, left: 60 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    const n = Math.max(fpr.length, tpr.length);

    // Build SVG polylines
    function toPoints(arr) {
        return arr.map((v, i) => {
            const x = pad.left + (i / Math.max(n - 1, 1)) * plotW;
            const y = pad.top + (1 - v) * plotH;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(' ');
    }

    // Find KS point (max gap)
    let ksIdx = 0, ksGap = 0;
    for (let i = 0; i < Math.min(fpr.length, tpr.length); i++) {
        const gap = Math.abs(tpr[i] - fpr[i]);
        if (gap > ksGap) { ksGap = gap; ksIdx = i; }
    }
    const ksX = pad.left + (ksIdx / Math.max(n - 1, 1)) * plotW;
    const ksY1 = pad.top + (1 - fpr[ksIdx]) * plotH;
    const ksY2 = pad.top + (1 - tpr[ksIdx]) * plotH;

    let svg = `<svg viewBox="0 0 ${W} ${H}" class="ks-chart" preserveAspectRatio="xMidYMid meet">`;

    // Grid lines
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (i / 4) * plotH;
        const val = (1 - i / 4);
        svg += `<line x1="${pad.left}" y1="${y}" x2="${pad.left + plotW}" y2="${y}" stroke="#e2e8f0" stroke-width="1"/>`;
        svg += `<text x="${pad.left - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#64748b">${val.toFixed(1)}</text>`;
    }

    // Axes
    svg += `<line x1="${pad.left}" y1="${pad.top + plotH}" x2="${pad.left + plotW}" y2="${pad.top + plotH}" stroke="#475569" stroke-width="2"/>`;
    svg += `<line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + plotH}" stroke="#475569" stroke-width="2"/>`;

    // Axis labels
    svg += `<text x="${pad.left + plotW / 2}" y="${H - 10}" text-anchor="middle" font-size="12" fill="#475569">\u7D2F\u8BA1\u4EBA\u7FA4\u6BD4\u4F8B</text>`;
    svg += `<text x="15" y="${pad.top + plotH / 2}" transform="rotate(-90 15 ${pad.top + plotH / 2})" text-anchor="middle" font-size="12" fill="#475569">\u7D2F\u8BA1\u6BD4\u4F8B</text>`;

    // FPR line (red)
    svg += `<polyline points="${toPoints(fpr)}" fill="none" stroke="#ef4444" stroke-width="2"/>`;
    // TPR line (green)
    svg += `<polyline points="${toPoints(tpr)}" fill="none" stroke="#10b981" stroke-width="2"/>`;

    // KS gap marker
    if (ksGap > 0) {
        svg += `<line x1="${ksX}" y1="${ksY1}" x2="${ksX}" y2="${ksY2}" stroke="#3b82f6" stroke-width="2" stroke-dasharray="5,3"/>`;
        svg += `<text x="${ksX + 5}" y="${(ksY1 + ksY2) / 2}" font-size="12" fill="#3b82f6" font-weight="bold">KS=${formatNumber(data.ks_max, 4)}</text>`;
    }

    // Legend
    svg += `<rect x="${pad.left + 10}" y="${pad.top + 5}" width="12" height="12" fill="#10b981"/>`;
    svg += `<text x="${pad.left + 28}" y="${pad.top + 15}" font-size="11" fill="#475569">TPR (\u597D\u5BA2\u6237\u7D2F\u8BA1)</text>`;
    svg += `<rect x="${pad.left + 10}" y="${pad.top + 22}" width="12" height="12" fill="#ef4444"/>`;
    svg += `<text x="${pad.left + 28}" y="${pad.top + 32}" font-size="11" fill="#475569">FPR (\u574F\u5BA2\u6237\u7D2F\u8BA1)</text>`;

    svg += '</svg>';
    container.innerHTML = svg;
}

/**
 * Render decile lift table.
 * @param {Array<object>} data - [{ decile, samples, bad, bad_rate, cum_bad_rate, lift, cum_lift }]
 */
function renderLiftTable(data) {
    const container = document.getElementById('lift-content');
    if (!container) return;
    if (!data || data.length === 0) {
        container.innerHTML = '<p class="text-muted">\u65E0 Lift \u6570\u636E</p>';
        return;
    }
    let html = '<table class="data-table"><thead><tr>'
        + '<th>\u5341\u5206\u4F4D</th><th>\u6837\u672C\u6570</th><th>\u574F\u5BA2\u6237</th>'
        + '<th>\u574F\u5BA2\u6237\u7387</th><th>\u7D2F\u8BA1\u574F\u5BA2\u7387</th><th>Lift</th><th>\u7D2F\u8BA1 Lift</th>'
        + '</tr></thead><tbody>';
    data.forEach(d => {
        const liftColor = d.lift >= 1 ? '#10b981' : '#6b7280';
        html += `<tr>`
            + `<td>${formatNumber(d.decile, 0)}</td>`
            + `<td>${formatNumber(d.samples, 0)}</td>`
            + `<td>${formatNumber(d.bad, 0)}</td>`
            + `<td>${formatPercent(d.bad_rate)}</td>`
            + `<td>${formatPercent(d.cum_bad_rate)}</td>`
            + `<td style="color:${liftColor};font-weight:bold;">${formatNumber(d.lift, 2)}</td>`
            + `<td>${formatNumber(d.cum_lift, 2)}</td>`
            + `</tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

/**
 * Render score distribution histogram using div elements.
 * @param {object} data - { bins: [...], counts: [...] }
 */
function renderScoreDist(data) {
    const container = document.getElementById('score-dist-content');
    if (!container) return;
    const bins = data.bins || [];
    const counts = data.counts || [];
    if (counts.length === 0) {
        container.innerHTML = '<p class="text-muted">\u65E0\u8BC4\u5206\u5206\u5E03\u6570\u636E</p>';
        return;
    }
    const maxCount = Math.max(...counts, 1);
    let html = '<div class="histogram">';
    for (let i = 0; i < counts.length; i++) {
        const height = (counts[i] / maxCount) * 100;
        const binLabel = bins[i] !== undefined ? formatNumber(bins[i], 2) : '';
        const nextLabel = bins[i + 1] !== undefined ? formatNumber(bins[i + 1], 2) : '';
        html += `<div class="histogram-col">`
            + `<div class="histogram-bar" style="height:${height}%;background-color:#3b82f6;" title="${binLabel}-${nextLabel}: ${counts[i]}">`
            + `<span class="histogram-count">${counts[i] > 0 ? counts[i] : ''}</span>`
            + `</div>`
            + `<div class="histogram-label">${binLabel}</div>`
            + `</div>`;
    }
    html += '</div>';
    container.innerHTML = html;
}

/* ============================================================
 * Screen: Copilot (AI Assistant)
 * ============================================================ */

/**
 * Send a message to the copilot and render the response.
 * Tries POST /api/v1/copilot/chat; falls back to local responses.
 */
async function sendCopilotMessage() {
    const input = document.getElementById('copilot-input');
    if (!input) return;
    const message = input.value.trim();
    if (!message) return;

    renderCopilotMessage(message, 'user');
    input.value = '';

    // Try API endpoint first
    try {
        const res = await api.post('/api/v1/copilot/chat', { message }, { silent: true });
        if (res && res.response) {
            renderCopilotMessage(res.response, 'bot');
            return;
        }
    } catch {
        // Fall through to local response
    }

    // Local fallback: generate a helpful response from current state
    const reply = _generateCopilotReply(message);
    renderCopilotMessage(reply, 'bot');
}

/**
 * Generate a local copilot response based on the current application state.
 * @param {string} message
 * @returns {string}
 */
function _generateCopilotReply(message) {
    const lower = message.toLowerCase();

    // Check for health/status
    if (lower.includes('健康') || lower.includes('状态') || lower.includes('status') || lower.includes('health')) {
        return state.apiOnline
            ? 'API \u670D\u52A1\u8FD0\u884C\u6B63\u5E38\uFF0C\u7248\u672C 0.1.0\u3002'
            : 'API \u670D\u52A1\u672A\u8FDE\u63A5\uFF0C\u8BF7\u68C0\u67E5\u540E\u7AEF\u662F\u5426\u542F\u52A8\u3002';
    }

    // Check for data status
    if (lower.includes('数据') || lower.includes('data') || lower.includes('样本')) {
        if (state.dataSummary) {
            const s = state.dataSummary;
            return `\u5F53\u524D\u6570\u636E\uFF1A${s.total_samples} \u6837\u672C\uFF0C${s.num_features} \u4E2A\u7279\u5F81\uFF0C\u597D\u5BA2\u6237 ${s.good_count}\uFF0C\u574F\u5BA2\u6237 ${s.bad_count}\u3002`;
        }
        return '\u5C1A\u672A\u52A0\u8F7D\u6570\u636E\uFF0C\u8BF7\u5728\u201C\u6570\u636E\u4E0A\u4F20\u201D\u9875\u9762\u52A0\u8F7D\u6837\u4F8B\u6570\u636E\u6216\u4E0A\u4F20\u6587\u4EF6\u3002';
    }

    // Check for model status
    if (lower.includes('模型') || lower.includes('model') || lower.includes('训练') || lower.includes('train')) {
        if (state.trainResult) {
            const r = state.trainResult;
            const tm = r.train_metrics || {};
            const em = r.test_metrics || {};
            return `\u6A21\u578B\u5DF2\u8BAD\u7EC3\uFF1A${r.model_type}\uFF0C${r.n_features} \u4E2A\u7279\u5F81\u3002\u8BAD\u7EC3\u96C6 AUC=${formatNumber(tm.auc, 4)}\u3001KS=${formatNumber(tm.ks, 4)}\uFF1B\u6D4B\u8BD5\u96C6 AUC=${formatNumber(em.auc, 4)}\u3001KS=${formatNumber(em.ks, 4)}\u3002`;
        }
        return '\u5C1A\u672A\u8BAD\u7EC3\u6A21\u578B\uFF0C\u8BF7\u5148\u5B8C\u6210\u6570\u636E\u52A0\u8F7D\u3001EDA\u3001\u5206\u7BB1\u540E\u6267\u884C\u8BAD\u7EC3\u3002';
    }

    // Check for workflow
    if (lower.includes('流程') || lower.includes('workflow') || lower.includes('步骤') || lower.includes('step')) {
        return '\u5EFA\u6A21\u6D41\u7A0B\uFF1A1. \u6570\u636E\u4E0A\u4F20 -> 2. EDA \u63A2\u7D22 -> 3. WOE/IV \u5206\u7BB1 -> 4. \u6A21\u578B\u8BAD\u7EC3 -> 5. \u6A21\u578B\u8BC4\u4F30 -> 6. \u5BFC\u51FA / \u90E8\u7F72 -> 7. \u76D1\u63A7 / \u53EF\u89E3\u91CA\u6027\u3002';
    }

    // Default
    return '\u6211\u662F\u98CE\u63A7\u5EFA\u6A21\u52A9\u624B\u3002\u60A8\u53EF\u4EE5\u8BE2\u95EE\u6570\u636E\u72B6\u6001\u3001\u6A21\u578B\u4FE1\u606F\u3001\u5EFA\u6A21\u6D41\u7A0B\u7B49\u95EE\u9898\u3002';
}

/**
 * Render a single chat message in the copilot conversation.
 * @param {string} text - Message text
 * @param {string} sender - 'user' | 'bot'
 */
function renderCopilotMessage(text, sender) {
    const container = document.getElementById('copilot-messages');
    if (!container) return;
    const msg = document.createElement('div');
    msg.className = `chat-message chat-message-${sender}`;
    msg.innerHTML = `<div class="chat-bubble">${escapeHTML(text)}</div>`;
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

/**
 * Clear the copilot chat history.
 */
function clearCopilotChat() {
    setContainerHTML('copilot-messages', '');
    renderCopilotMessage('\u6B22\u8FCE\u4F7F\u7528\u98CE\u63A7\u5EFA\u6A21\u52A9\u624B\uFF0C\u8BF7\u8F93\u5165\u60A8\u7684\u95EE\u9898\u3002', 'bot');
}

/* ============================================================
 * Screen: Export
 * ============================================================ */

/**
 * Export the model as an HTML report.
 * Calls POST /api/v1/export/html, opens result in a new tab.
 */
async function exportHTML() {
    try {
        const html = await api.postText('/api/v1/export/html');
        if (html) {
            const blob = new Blob([html], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            window.open(url, '_blank');
            setTimeout(() => URL.revokeObjectURL(url), 5000);
            showToast('HTML \u62A5\u544A\u5DF2\u751F\u6210\u5E76\u5728\u65B0\u6807\u7B7E\u9875\u6253\u5F00', 'success');
        } else {
            showToast('\u5BFC\u51FA HTML \u5931\u8D25', 'error');
        }
    } catch (error) {
        showToast(`\u5BFC\u51FA HTML \u5931\u8D25: ${error.message}`, 'error');
    }
}

/**
 * Export the model as a Python scoring script.
 * Calls POST /api/v1/export/python, triggers file download.
 */
async function exportPython() {
    try {
        const blob = await api.postBlob('/api/v1/export/python');
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'scorecard.py';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 3000);
        showToast('Python \u811A\u672C\u5DF2\u4E0B\u8F7D', 'success');
    } catch (error) {
        showToast(`\u5BFC\u51FA Python \u5931\u8D25: ${error.message}`, 'error');
    }
}

/* ============================================================
 * Screen: Deploy
 * ============================================================ */

/**
 * Display API usage examples for the deployed scoring service.
 * Fetches model info and generates curl / Python examples.
 */
async function showAPIExample() {
    const container = document.getElementById('deploy-api-example');
    if (!container) return;

    let features = state.trainResult?.features || [];
    let baseScore = state.trainResult?.base_score || 600;

    // Try to get model info from API
    try {
        const info = await api.get('/api/v1/model/info', { silent: true });
        if (info && info.n_features !== undefined) {
            features = features.length > 0 ? features : ['feature1', 'feature2'];
            baseScore = info.base_score || baseScore;
        }
    } catch {
        // Use state data
        if (features.length === 0) {
            features = ['checking_status', 'duration', 'credit_amount'];
        }
    }

    // Build sample feature JSON
    const sampleFeatures = {};
    features.slice(0, 5).forEach(f => {
        sampleFeatures[f] = 'value';
    });
    const sampleJSON = JSON.stringify(sampleFeatures, null, 2);

    const apiURL = `${CONFIG.apiBase}/api/v1/score`;

    const curlExample = `# cURL\n`
        + `curl -X POST ${apiURL} \\\n`
        + `  -H "Content-Type: application/json" \\\n`
        + `  -d '${JSON.stringify(sampleFeatures)}'`;

    const pythonExample = `# Python\n`
        + `import requests\n\n`
        + `url = "${apiURL}"\n`
        + `payload = ${sampleJSON}\n\n`
        + `resp = requests.post(url, json=payload)\n`
        + `result = resp.json()\n`
        + `print(f"Score: {result['score']}, Risk: {result['risk_level']}")`;

    const batchExample = `# Batch Scoring\n`
        + `curl -X POST ${CONFIG.apiBase}/api/v1/score/batch \\\n`
        + `  -H "Content-Type: application/json" \\\n`
        + `  -d '{"records": [${JSON.stringify(sampleFeatures)}, ${JSON.stringify(sampleFeatures)}]}'`;

    let html = '<div class="code-sections">';
    html += `<div class="code-block"><div class="code-header">cURL \u5355\u6761\u8BC4\u5206</div><pre>${escapeHTML(curlExample)}</pre></div>`;
    html += `<div class="code-block"><div class="code-header">Python \u793A\u4F8B</div><pre>${escapeHTML(pythonExample)}</pre></div>`;
    html += `<div class="code-block"><div class="code-header">cURL \u6279\u91CF\u8BC4\u5206</div><pre>${escapeHTML(batchExample)}</pre></div>`;
    html += '</div>';

    container.innerHTML = html;
    showToast('API \u8C03\u7528\u793A\u4F8B\u5DF2\u751F\u6210', 'success');
}

/**
 * Display SQL rules derived from the scorecard.
 * Fetches the scorecard table and generates CASE WHEN statements.
 */
async function showSQLExport() {
    const container = document.getElementById('deploy-sql-export');
    if (!container) return;

    let scorecard = state.scorecard;

    // Try to fetch from API if not in state
    if (!scorecard) {
        try {
            const res = await api.get('/api/v1/scorecard/table', { silent: true });
            if (res.scorecard) {
                scorecard = res.scorecard;
                state.scorecard = scorecard;
            }
        } catch {
            // ignore
        }
    }

    if (!scorecard || scorecard.length === 0) {
        container.innerHTML = '<p class="text-muted">\u8BF7\u5148\u8BAD\u7EC3\u6A21\u578B\u4EE5\u751F\u6210 SQL \u89C4\u5219</p>';
        return;
    }

    const baseScore = state.trainResult?.base_score || 600;

    let sql = `-- \u4FE1\u7528\u8BC4\u5206\u5361 SQL \u89C4\u5219\n-- \u81EA\u52A8\u751F\u6210\uFF0C\u53EF\u76F4\u63A5\u5D4C\u5165\u51B3\u7B56\u5F15\u64CE\n\n`;
    sql += `-- \u8BC4\u5206\u903B\u8F91\u8BF4\u660E:\n`;
    scorecard.slice(0, 10).forEach(item => {
        sql += `-- ${item.feature}: ${item.bin} -> ${item.score >= 0 ? '+' : ''}${item.score.toFixed(1)}\n`;
    });
    if (scorecard.length > 10) sql += `-- ... \u5171 ${scorecard.length} \u6761\u89C4\u5219\n`;
    sql += '\n';
    sql += `SELECT\n  ${baseScore} -- base_score\n`;
    scorecard.forEach(item => {
        const score = item.score.toFixed(1);
        sql += `  + CASE WHEN ${item.feature} = '${item.bin}' THEN ${score} ELSE 0 END\n`;
    });
    sql += `  AS credit_score\n`;
    sql += `FROM customer_applications;`;

    container.innerHTML = `<div class="code-block"><div class="code-header">SQL \u89C4\u5219</div><pre>${escapeHTML(sql)}</pre></div>`;
    showToast('SQL \u89C4\u5219\u5DF2\u751F\u6210', 'success');
}

/* ============================================================
 * Screen: Monitor
 * ============================================================ */

/**
 * Render the monitoring dashboard with mock PSI data and alerts.
 * Uses feature names from the trained model if available.
 */
function renderMonitorDashboard() {
    const container = document.getElementById('monitor-dashboard');
    if (!container) return;

    // Use real feature names if available, otherwise use defaults
    let features = [];
    if (state.trainResult && state.trainResult.features) {
        features = state.trainResult.features.slice(0, 8);
    }
    if (features.length === 0) {
        features = ['checking_status', 'duration', 'credit_amount', 'age', 'savings_status', 'employment', 'installment_commitment', 'purpose'];
    }

    // Generate mock PSI data
    const mockPSI = features.map((f, i) => {
        const psiValues = [0.04, 0.08, 0.12, 0.05, 0.28, 0.07, 0.15, 0.03];
        const psi = psiValues[i % psiValues.length];
        const status = psi < 0.1 ? 'stable' : psi < 0.25 ? 'warning' : 'unstable';
        return { feature: f, psi, status };
    });

    // Summary
    const stable = mockPSI.filter(p => p.status === 'stable').length;
    const warning = mockPSI.filter(p => p.status === 'warning').length;
    const unstable = mockPSI.filter(p => p.status === 'unstable').length;

    let html = '<div class="stat-grid">';
    html += _statCard('\u76D1\u63A7\u7279\u5F81\u6570', formatNumber(mockPSI.length, 0), '#3b82f6');
    html += _statCard('\u7A33\u5B9A', formatNumber(stable, 0), '#10b981');
    html += _statCard('\u8B66\u544A', formatNumber(warning, 0), '#f59e0b');
    html += _statCard('\u4E0D\u7A33\u5B9A', formatNumber(unstable, 0), '#ef4444');
    html += '</div>';

    // PSI table
    html += '<h4>\u7279\u5F81 PSI \u76D1\u63A7</h4>';
    html += '<table class="data-table"><thead><tr><th>\u7279\u5F81</th><th>PSI</th><th>\u72B6\u6001</th><th>\u8D8B\u52BF</th></tr></thead><tbody>';
    mockPSI.forEach(p => {
        const color = colorPSI(p.psi);
        const bar = renderBar(p.psi, 0.3, color);
        const statusText = p.status === 'stable' ? '\u7A33\u5B9A' : p.status === 'warning' ? '\u8B66\u544A' : '\u4E0D\u7A33\u5B9A';
        html += `<tr>`
            + `<td>${escapeHTML(p.feature)}</td>`
            + `<td style="color:${color};font-weight:bold;">${formatNumber(p.psi, 4)}</td>`
            + `<td><span class="status-badge" style="background-color:${color}22;color:${color};">${statusText}</span></td>`
            + `<td style="min-width:200px;">${bar}</td>`
            + `</tr>`;
    });
    html += '</tbody></table>';

    // Alerts
    const alerts = mockPSI.filter(p => p.status !== 'stable');
    if (alerts.length > 0) {
        html += '<h4>\u544A\u8B66\u5217\u8868</h4><div class="alert-list">';
        alerts.forEach(a => {
            const color = colorPSI(a.psi);
            const level = a.status === 'unstable' ? 'alert-danger' : 'alert-warning';
            const msg = a.status === 'unstable'
                ? `\u7279\u5F81 "${a.feature}" PSI=${formatNumber(a.psi, 4)}\uFF0C\u8D85\u8FC7 0.25 \u9608\u503C\uFF0C\u5EFA\u8BAE\u91CD\u65B0\u8BAD\u7EC3\u6A21\u578B\u3002`
                : `\u7279\u5F81 "${a.feature}" PSI=${formatNumber(a.psi, 4)}\uFF0C\u63A5\u8FD1\u9608\u503C\uFF0C\u5EFA\u8BAE\u5173\u6CE8\u3002`;
            html += `<div class="${level}" style="border-left:4px solid ${color};">${escapeHTML(msg)}</div>`;
        });
        html += '</div>';
    } else {
        html += '<div class="alert-success">\u6240\u6709\u7279\u5F81 PSI \u6B63\u5E38\uFF0C\u6A21\u578B\u8FD0\u884C\u7A33\u5B9A\u3002</div>';
    }

    // Mock score distribution comparison
    html += '<h4>\u8BC4\u5206\u5206\u5E03\u5BF9\u6BD4 (\u57FA\u51C6 vs \u5F53\u524D)</h4>';
    html += '<div class="histogram">';
    const baseDist = [5, 12, 25, 30, 18, 7, 3];
    const currDist = [4, 10, 20, 28, 22, 12, 4];
    const maxVal = Math.max(...baseDist, ...currDist, 1);
    for (let i = 0; i < baseDist.length; i++) {
        const h1 = (baseDist[i] / maxVal) * 100;
        const h2 = (currDist[i] / maxVal) * 100;
        html += `<div class="histogram-col">`
            + `<div class="histogram-bar-group">`
            + `<div class="histogram-bar" style="height:${h1}%;background-color:#94a3b8;width:40%;display:inline-block;" title="Base: ${baseDist[i]}%"></div>`
            + `<div class="histogram-bar" style="height:${h2}%;background-color:#3b82f6;width:40%;display:inline-block;margin-left:4px;" title="Current: ${currDist[i]}%"></div>`
            + `</div>`
            + `<div class="histogram-label">${i + 1}</div>`
            + `</div>`;
    }
    html += '</div>';
    html += '<div style="margin-top:8px;"><span style="display:inline-block;width:12px;height:12px;background:#94a3b8;margin-right:4px;"></span>\u57FA\u51C6\u671F &nbsp; <span style="display:inline-block;width:12px;height:12px;background:#3b82f6;margin-right:4px;"></span>\u5F53\u524D\u671F</div>';

    container.innerHTML = html;
    showToast('\u76D1\u63A7\u9762\u677F\u5DF2\u52A0\u8F7D', 'info');
}

/* ============================================================
 * Screen: Explain (Model Interpretability)
 * ============================================================ */

/**
 * Load algorithm justification.
 * Calls GET /api/v1/explain/algorithm
 */
async function loadAlgorithmJustification() {
    const container = document.getElementById('explain-algorithm');
    if (!container) return;
    try {
        const data = await api.get('/api/v1/explain/algorithm');
        let html = `<div class="info-block">`;
        html += `<h4>${escapeHTML(data.selected_algorithm || '')}</h4>`;
        html += `<p>${escapeHTML(data.reason || '')}</p>`;
        html += `</div>`;

        // Comparison table
        if (data.comparison && data.comparison.length > 0) {
            html += '<h4>\u7B97\u6CD5\u5BF9\u6BD4</h4>';
            html += '<table class="data-table"><thead><tr><th>\u7EF4\u5EA6</th><th>LR + WOE</th><th>XGBoost</th><th>\u80DC\u51FA</th><th>\u539F\u56E0</th></tr></thead><tbody>';
            data.comparison.forEach(c => {
                const winnerColor = c.winner === 'LR' ? '#10b981' : '#ef4444';
                html += `<tr>`
                    + `<td>${escapeHTML(c.dimension || '')}</td>`
                    + `<td>${escapeHTML(c.lr || '')}</td>`
                    + `<td>${escapeHTML(c.xgboost || '')}</td>`
                    + `<td style="color:${winnerColor};font-weight:bold;">${escapeHTML(c.winner || '')}</td>`
                    + `<td>${escapeHTML(c.reason || '')}</td>`
                    + `</tr>`;
            });
            html += '</tbody></table>';
        }

        // Conclusion
        if (data.conclusion) {
            html += `<div class="info-block"><h4>\u7ED3\u8BBA</h4><p>${escapeHTML(data.conclusion)}</p></div>`;
        }

        // WOE reasoning
        if (data.woe_reasoning) {
            const wr = data.woe_reasoning;
            html += `<div class="info-block"><h4>WOE \u7F16\u7801\u9009\u62E9</h4><p>${escapeHTML(wr.reason || '')}</p></div>`;
            if (wr.comparison && wr.comparison.length > 0) {
                html += '<table class="data-table"><thead><tr><th>\u65B9\u6CD5</th><th>\u53EF\u89E3\u91CA\u6027</th><th>\u5355\u8C03\u6027</th><th>\u7F3A\u5931\u503C</th><th>\u9C81\u68D2\u6027</th></tr></thead><tbody>';
                wr.comparison.forEach(c => {
                    html += `<tr>`
                        + `<td>${escapeHTML(c.method || '')}</td>`
                        + `<td>${escapeHTML(c.explainability || '')}</td>`
                        + `<td>${escapeHTML(c.monotonicity || '')}</td>`
                        + `<td>${escapeHTML(c.missing || '')}</td>`
                        + `<td>${escapeHTML(c.robust || '')}</td>`
                        + `</tr>`;
                });
                html += '</tbody></table>';
            }
        }

        container.innerHTML = html;
        showToast('\u7B97\u6CD5\u8BBA\u8BC1\u5DF2\u52A0\u8F7D', 'success');
    } catch (error) {
        container.innerHTML = `<p class="text-muted">\u52A0\u8F7D\u5931\u8D25: ${escapeHTML(error.message)}</p>`;
        showToast(`\u52A0\u8F7D\u5931\u8D25: ${error.message}`, 'error');
    }
}

/**
 * Load feature importance ranking.
 * Calls GET /api/v1/explain/importance
 */
async function loadFeatureImportance() {
    const container = document.getElementById('explain-importance');
    if (!container) return;
    try {
        const data = await api.get('/api/v1/explain/importance');
        const importance = data.importance || [];
        if (importance.length === 0) {
            container.innerHTML = '<p class="text-muted">\u65E0\u7279\u5F81\u91CD\u8981\u6027\u6570\u636E</p>';
            return;
        }
        const maxPct = Math.max(...importance.map(i => i.importance_pct || 0), 1);
        let html = '<table class="data-table"><thead><tr><th>\u7279\u5F81</th><th>\u7CFB\u6570</th><th>|\u7CFB\u6570|</th><th>\u91CD\u8981\u6027</th></tr></thead><tbody>';
        importance.forEach(item => {
            const color = item.coef >= 0 ? '#10b981' : '#ef4444';
            const bar = renderBar(item.importance_pct, maxPct, '#3b82f6');
            html += `<tr>`
                + `<td>${escapeHTML(item.feature)}</td>`
                + `<td style="color:${color};">${formatNumber(item.coef, 6)}</td>`
                + `<td>${formatNumber(item.abs_coef, 6)}</td>`
                + `<td style="min-width:200px;"><div style="display:flex;align-items:center;gap:8px;">${bar}<span>${formatNumber(item.importance_pct, 2)}%</span></div></td>`
                + `</tr>`;
        });
        html += '</tbody></table>';
        container.innerHTML = html;
        showToast('\u7279\u5F81\u91CD\u8981\u6027\u5DF2\u52A0\u8F7D', 'success');
    } catch (error) {
        container.innerHTML = `<p class="text-muted">\u52A0\u8F7D\u5931\u8D25: ${escapeHTML(error.message)}</p>`;
        showToast(`\u52A0\u8F7D\u5931\u8D25: ${error.message}`, 'error');
    }
}

/**
 * Explain a single sample prediction.
 * Calls GET /api/v1/explain/sample/{index}
 * @param {number} index - Sample index (if omitted, reads from input)
 */
async function explainSample(index) {
    const container = document.getElementById('explain-sample');
    if (!container) return;

    // Read index from input if not provided
    if (index === undefined || index === null) {
        const input = document.getElementById('explain-sample-index');
        if (!input || !input.value) {
            showToast('\u8BF7\u8F93\u5165\u6837\u672C\u7D22\u5F15', 'warning');
            return;
        }
        index = parseInt(input.value, 10);
        if (isNaN(index) || index < 0) {
            showToast('\u8BF7\u8F93\u5165\u6709\u6548\u7684\u6837\u672C\u7D22\u5F15', 'warning');
            return;
        }
    }

    try {
        const data = await api.get(`/api/v1/explain/sample/${index}`);

        let html = '<div class="stat-grid">';
        const scoreColor = colorScore(data.score || 0);
        html += _statCard('\u8BC4\u5206', formatNumber(data.score, 0), scoreColor);
        html += _statCard('\u9608\u503C', formatNumber(data.threshold, 0), '#6b7280');
        const decisionColor = data.decision === 'approve' ? '#10b981' : '#ef4444';
        const decisionText = data.decision === 'approve' ? '\u901A\u8FC7' : '\u62D2\u7EDD';
        html += _statCard('\u51B3\u7B56', decisionText, decisionColor);
        html += '</div>';

        // Reason text
        if (data.reason_text) {
            html += `<div class="info-block"><p>${escapeHTML(data.reason_text)}</p></div>`;
        }

        // Sample data
        if (data.sample_data) {
            html += '<h4>\u6837\u672C\u6570\u636E</h4>';
            html += '<table class="data-table"><thead><tr><th>\u5B57\u6BB5</th><th>\u503C</th></tr></thead><tbody>';
            for (const [k, v] of Object.entries(data.sample_data)) {
                html += `<tr><td>${escapeHTML(k)}</td><td>${escapeHTML(String(v))}</td></tr>`;
            }
            html += '</tbody></table>';
        }

        // Breakdown
        if (data.breakdown && data.breakdown.length > 0) {
            html += '<h4>\u7279\u5F81\u8D21\u732E\u5206\u89E3</h4>';
            html += '<table class="data-table"><thead><tr><th>\u7279\u5F81</th><th>\u503C</th><th>\u7CFB\u6570</th><th>\u8D21\u732E</th><th>\u65B9\u5411</th></tr></thead><tbody>';
            data.breakdown.forEach(b => {
                const dirColor = b.direction === 'positive' ? '#10b981' : '#ef4444';
                const dirText = b.direction === 'positive' ? '\u6B63\u5411' : '\u8D1F\u5411';
                html += `<tr>`
                    + `<td>${escapeHTML(b.feature)}</td>`
                    + `<td>${formatNumber(b.value, 4)}</td>`
                    + `<td>${formatNumber(b.coef, 6)}</td>`
                    + `<td style="color:${dirColor};">${formatNumber(b.contribution, 4)}</td>`
                    + `<td><span class="status-badge" style="background-color:${dirColor}22;color:${dirColor};">${dirText}</span></td>`
                    + `</tr>`;
            });
            html += '</tbody></table>';
        }

        // Top negative factors
        if (data.top_negative_factors && data.top_negative_factors.length > 0) {
            html += '<h4>\u4E3B\u8981\u98CE\u9669\u56E0\u7D20</h4><div class="tag-list">';
            data.top_negative_factors.forEach(f => {
                html += `<span class="tag tag-danger">${escapeHTML(f.feature)} (${formatNumber(f.contribution, 4)})</span>`;
            });
            html += '</div>';
        }

        container.innerHTML = html;
        showToast(`\u6837\u672C #${index} \u89E3\u91CA\u5DF2\u52A0\u8F7D`, 'success');
    } catch (error) {
        container.innerHTML = `<p class="text-muted">\u89E3\u91CA\u5931\u8D25: ${escapeHTML(error.message)}</p>`;
        showToast(`\u89E3\u91CA\u5931\u8D25: ${error.message}`, 'error');
    }
}

/**
 * Load the compliance audit report.
 * Calls GET /api/v1/explain/audit
 */
async function loadAuditReport() {
    const container = document.getElementById('explain-audit');
    if (!container) return;
    try {
        const data = await api.get('/api/v1/explain/audit');

        let html = '<div class="stat-grid">';
        const overallColor = data.overall_status === 'compliant' ? '#10b981' : '#f59e0b';
        const overallText = data.overall_status === 'compliant' ? '\u5408\u89C4' : '\u9700\u5173\u6CE8';
        html += _statCard('\u603B\u68C0\u67E5\u9879', formatNumber(data.total_checks, 0), '#3b82f6');
        html += _statCard('\u901A\u8FC7', formatNumber(data.passed, 0), '#10b981');
        html += _statCard('\u8B66\u544A', formatNumber(data.warnings, 0), '#f59e0b');
        html += _statCard('\u603B\u4F53\u72B6\u6001', overallText, overallColor);
        html += '</div>';

        if (data.checks && data.checks.length > 0) {
            html += '<h4>\u5BA1\u8BA1\u68C0\u67E5\u660E\u7EC6</h4>';
            html += '<table class="data-table"><thead><tr><th>\u68C0\u67E5\u9879</th><th>\u8981\u6C42</th><th>\u72B6\u6001</th><th>\u8BE6\u60C5</th></tr></thead><tbody>';
            data.checks.forEach(c => {
                const statusColor = c.status === 'pass' ? '#10b981' : '#f59e0b';
                const statusText = c.status === 'pass' ? '\u901A\u8FC7' : '\u8B66\u544A';
                html += `<tr>`
                    + `<td>${escapeHTML(c.item)}</td>`
                    + `<td>${escapeHTML(c.requirement)}</td>`
                    + `<td><span class="status-badge" style="background-color:${statusColor}22;color:${statusColor};">${statusText}</span></td>`
                    + `<td>${escapeHTML(c.detail)}</td>`
                    + `</tr>`;
            });
            html += '</tbody></table>';
        }

        container.innerHTML = html;
        showToast('\u5BA1\u8BA1\u62A5\u544A\u5DF2\u52A0\u8F7D', 'success');
    } catch (error) {
        container.innerHTML = `<p class="text-muted">\u52A0\u8F7D\u5931\u8D25: ${escapeHTML(error.message)}</p>`;
        showToast(`\u52A0\u8F7D\u5931\u8D25: ${error.message}`, 'error');
    }
}

/* ============================================================
 * Event Listeners Setup
 * ============================================================ */

/**
 * Register all DOM event listeners after the page is ready.
 */
function setupEventListeners() {
    // Navigation items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const screen = item.dataset.screen;
            if (screen) switchScreen(screen);
        });
    });

    // --- Upload screen ---
    const btnLoadSample = document.getElementById('btn-load-sample');
    if (btnLoadSample) btnLoadSample.addEventListener('click', loadSample);

    const btnChooseFile = document.getElementById('btn-choose-file');
    if (btnChooseFile) btnChooseFile.addEventListener('click', () => {
        document.getElementById('file-input').click();
    });

    const btnUploadFile = document.getElementById('btn-upload-file');
    if (btnUploadFile) btnUploadFile.addEventListener('click', uploadFile);

    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.addEventListener('change', () => {
        const label = document.getElementById('file-name-display');
        if (label && fileInput.files.length > 0) {
            label.style.display = 'block';
            label.textContent = '已选择: ' + fileInput.files[0].name;
        }
        // Auto upload after file selection
        if (fileInput.files.length > 0) {
            uploadFile();
        }
    });

    // Upload zone click to select file
    const uploadZone = document.getElementById('upload-zone');
    if (uploadZone) {
        uploadZone.addEventListener('click', (e) => {
            // Don't trigger if clicking the button itself
            if (e.target.tagName !== 'BUTTON') {
                document.getElementById('file-input').click();
            }
        });
        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('active');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('active');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('active');
            if (e.dataTransfer.files.length > 0) {
                const fileInput = document.getElementById('file-input');
                // Create a new FileList is tricky, so we use DataTransfer
                const dt = new DataTransfer();
                dt.items.add(e.dataTransfer.files[0]);
                fileInput.files = dt.files;
                // Trigger change event
                fileInput.dispatchEvent(new Event('change'));
            }
        });
    }

    // --- EDA screen ---
    const btnRunEDA = document.getElementById('btn-run-eda');
    if (btnRunEDA) btnRunEDA.addEventListener('click', runEDA);

    // --- Binning screen ---
    var ss = document.getElementById('split-ratio-slider');
    var sd = document.getElementById('split-ratio-display');
    if (ss) { ss.addEventListener('input', function() { var t = parseInt(this.value); if (sd) sd.textContent = (t/10) + ' : ' + ((100-t)/10); }); }
    var ss = document.getElementById('split-ratio-slider');
    var sd = document.getElementById('split-ratio-display');
    if (ss) { ss.addEventListener('input', function() { var t = parseInt(this.value); if (sd) sd.textContent = (t/10) + ' : ' + ((100-t)/10); }); }
    const btnRunBinning = document.getElementById('btn-run-binning');
    if (btnRunBinning) btnRunBinning.addEventListener('click', runBinning);

    // --- Training screen ---
    const btnRunTraining = document.getElementById('btn-run-training');
    if (btnRunTraining) btnRunTraining.addEventListener('click', runTraining);

    // --- Evaluation screen ---
    const btnRefreshEval = document.getElementById('btn-refresh-eval');
    if (btnRefreshEval) btnRefreshEval.addEventListener('click', async () => {
        try {
            const res = await api.get('/api/v1/evaluation/result');
            state.evalResult = res;
            renderEvaluation(res);
            showToast('\u8BC4\u4F30\u6570\u636E\u5DF2\u5237\u65B0', 'success');
        } catch (error) {
            showToast(`\u52A0\u8F7D\u8BC4\u4F30\u6570\u636E\u5931\u8D25: ${error.message}`, 'error');
        }
    });

    // --- Copilot screen ---
    const btnCopilotSend = document.getElementById('btn-copilot-send');
    if (btnCopilotSend) btnCopilotSend.addEventListener('click', sendCopilotMessage);

    const copilotInput = document.getElementById('copilot-input');
    if (copilotInput) copilotInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendCopilotMessage();
        }
    });

    const btnCopilotClear = document.getElementById('btn-copilot-clear');
    if (btnCopilotClear) btnCopilotClear.addEventListener('click', clearCopilotChat);

    // --- Export screen ---
    const btnExportHTML = document.getElementById('btn-export-html');
    if (btnExportHTML) btnExportHTML.addEventListener('click', exportHTML);

    const btnExportPython = document.getElementById('btn-export-python');
    if (btnExportPython) btnExportPython.addEventListener('click', exportPython);

    // --- Deploy screen ---
    const btnShowAPI = document.getElementById('btn-show-api');
    if (btnShowAPI) btnShowAPI.addEventListener('click', showAPIExample);

    const btnShowSQL = document.getElementById('btn-show-sql');
    if (btnShowSQL) btnShowSQL.addEventListener('click', showSQLExport);

    // --- Monitor screen ---
    const btnRenderMonitor = document.getElementById('btn-render-monitor');
    if (btnRenderMonitor) btnRenderMonitor.addEventListener('click', renderMonitorDashboard);

    // --- Explain screen ---
    const btnLoadAlgorithm = document.getElementById('btn-load-algorithm');
    if (btnLoadAlgorithm) btnLoadAlgorithm.addEventListener('click', loadAlgorithmJustification);

    const btnLoadImportance = document.getElementById('btn-load-importance');
    if (btnLoadImportance) btnLoadImportance.addEventListener('click', loadFeatureImportance);

    const btnExplainSample = document.getElementById('btn-explain-sample');
    if (btnExplainSample) btnExplainSample.addEventListener('click', () => explainSample());

    const btnLoadAudit = document.getElementById('btn-load-audit');
    if (btnLoadAudit) btnLoadAudit.addEventListener('click', loadAuditReport);

    var btnRunAll = document.getElementById('btn-run-all');
    if (btnRunAll) btnRunAll.addEventListener('click', runAllSteps);
}

async function runAllSteps() {
    try {
        if (!state.dataSummary) {
            showToast('正在加载数据...', 'info');
            try {
                var r = await api.post('/api/v1/data/load-sample');
                if (r.status === 'ok') {
                    state.dataSummary = r.summary;
                    state.dataPreview = r.preview;
                    displaySummary(r);
                    showToast('数据加载完成', 'success');
                } else { showToast('请先上传数据', 'error'); return; }
            } catch(e) { showToast('请先上传数据', 'error'); return; }
        }
        switchScreen('eda');
        showToast('开始 EDA 分析...', 'info');
        var er = await api.post('/api/v1/eda/run');
        if (er.status === 'ok') { state.edaResult = er.result; _renderAllEDA(er.result); showToast('EDA 完成', 'success'); }
        else { showToast('EDA 失败', 'error'); return; }
        switchScreen('binning');
        showToast('开始分箱...', 'info');
        var br = await api.post('/api/v1/binning/run');
        if (br.status === 'ok') { state.binningResult = br.result; _renderBinningSummary(br.result); renderIVRanking(); showToast('分箱完成', 'success'); }
        else { showToast('分箱失败', 'error'); return; }
        switchScreen('training');
        showToast('开始训练...', 'info');
        var tr = await api.post('/api/v1/training/run');
        if (tr.status === 'ok') {
            state.trainResult = tr.result;
            state.evalResult = tr.evaluation || tr.result.train_metrics;
            renderModelResult(tr.result);
            try {
                var scr = await api.get('/api/v1/scorecard/table');
                if (scr.scorecard) { state.scorecard = scr.scorecard; renderScorecard(scr.scorecard); }
            } catch(se) { if (tr.result.scorecard_preview) { state.scorecard = tr.result.scorecard_preview; renderScorecard(tr.result.scorecard_preview); } }
            showToast('训练完成', 'success');
        } else { showToast('训练失败', 'error'); return; }
        switchScreen('evaluation');
        if (state.evalResult) { try { renderEvaluation(state.evalResult); } catch(ee) {} }
        showToast('一键建模完成！', 'success');
    } catch(error) { showToast('一键建模失败: ' + error.message, 'error'); }
}

/* ============================================================
 * Initialization
 * ============================================================ */

/**
 * Initialize the application on page load.
 * - Check API health
 * - Set up event listeners
 * - Switch to default screen
 * - Initialize copilot greeting
 */
// SQL export placeholder
function exportSQL() {
    showToast('SQL导出功能开发中', 'info');
}

// PSI calculation placeholder
function calcPSI() {
    showToast('PSI计算功能开发中', 'info');
}

// Global App object for inline HTML handlers
window.App = {
    switchScreen: switchScreen,
    toast: showToast,
    exportHTML: exportHTML,
    exportPython: exportPython,
    exportSQL: exportSQL,
    calcPSI: calcPSI,
    explainSample: explainSample,
    uploadFile: uploadFile,
    runAllSteps: runAllSteps,
};

async function init() {
    console.log('[Risk Modeling App] Initializing...');

    // Instantiate API client
    window.api = new RiskModelAPI(CONFIG.apiBase);

    // Set up event listeners
    setupEventListeners();

    // Check API health
    const isOnline = await window.api.checkHealth();
    if (isOnline) {
        showToast('API \u8FDE\u63A5\u6210\u529F', 'success', 2000);
    } else {
        showToast('API \u672A\u8FDE\u63A5\uFF0C\u8BF7\u786E\u4FDD\u540E\u7AEF\u670D\u52A1\u5DF2\u542F\u52A8', 'warning', 5000);
    }

    // Initialize copilot greeting
    const copilotMessages = document.getElementById('copilot-messages');
    if (copilotMessages && copilotMessages.children.length === 0) {
        renderCopilotMessage('\u6B22\u8FCE\u4F7F\u7528\u98CE\u63A7\u5EFA\u6A21\u52A9\u624B\u3002\u6D41\u7A0B\uFF1A\u6570\u636E\u4E0A\u4F20 -> EDA -> \u5206\u7BB1 -> \u8BAD\u7EC3 -> \u8BC4\u4F30 -> \u5BFC\u51FA\u3002\u8BF7\u8F93\u5165\u60A8\u7684\u95EE\u9898\u3002', 'bot');
    }

    // Switch to default screen
    switchScreen(CONFIG.defaultScreen);

    console.log('[Risk Modeling App] Initialization complete.');
}

/* ============================================================
 * Bootstrap
 * ============================================================ */

// Ensure the init function is called once the DOM is fully parsed.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    // DOMContentLoaded already fired (e.g. script loaded with defer)
    init();
}
