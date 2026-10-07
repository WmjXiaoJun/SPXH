/* SPXH 射频信号分析工作台 — 前端应用（原生 ES2020，无构建、无 CDN 依赖）
 * 契约：docs/前端接口约定-v1.md（字段名严格对齐）
 * 约定：本文件不使用模板字符串，避免与构建/注入工具冲突。
 */
(function () {
  'use strict';

  /* 未捕获异常/未处理的 Promise 拒绝一律送到错误条：
     否则前端一旦抛异常，页面就"静静地卡住"，用户只能看到加载占位符却不知道发生了什么 */
  if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('error', function (event) {
      try {
        var where = event && event.filename ? (event.filename.split('/').pop() + ':' + (event.lineno || 0)) : '';
        pushFrontendError('JS 异常：' + ((event && event.message) ? event.message : '未知错误') + (where ? (' @ ' + where) : ''));
      } catch (ignore) { /* 记录错误本身不能再抛 */ }
    });
    window.addEventListener('unhandledrejection', function (event) {
      try {
        var reason = event && event.reason;
        pushFrontendError('未处理的 Promise 拒绝：' + ((reason && reason.message) ? reason.message : String(reason)));
      } catch (ignore) { /* 同上 */ }
    });
  }

  function pushFrontendError(message) {
    if (typeof recordError !== 'function') { return; }
    recordError('前端', 0, message, '');
  }

  /* ================= 常量 ================= */
  var API_PREFIX = '/api';
  var MOCK_PREFIX = '/static/mock';
  var TABS = ['spectrum', 'estimate', 'features', 'classify', 'demod', 'fec', 'frame', 'ssca', 'narrate', 'agent', 'settings'];
  var TAB_LABEL = { spectrum: '频谱/瀑布', estimate: '参数估计', features: '特征', classify: '识别', demod: '解调', fec: '译码', frame: '帧', ssca: '谱相关', narrate: '报告', agent: '代理', settings: '设置' };

  /* /api/demod 的调制参数：'' = 自动（用后端分类结果），其余为手动指定，原样作为 modulation 查询参数 */
  var DEMOD_MODS = ['bpsk', 'qpsk', '8psk', '16qam', '32qam', '64qam', '2fsk', '4fsk', '8fsk'];
  var DEMOD_SOURCE_TEXT = { classifier: '来源 分类器', provided: '来源 手动指定', truth: '来源 真值' };

  var GROUP_LABELS = {
    cumulant: '累积量',
    spectral: '频谱',
    timefreq: '时频',
    constellation: '星座',
    cyclostationary: '循环平稳'
  };
  var GROUP_ORDER = ['cumulant', 'spectral', 'timefreq', 'constellation', 'cyclostationary'];

  /* 后端 /api/features 的 names/groups 缺失时的兜底目录（23 维，域划分与契约一致）。 */
  var FALLBACK_FEATURE_GROUPS = {
    cumulant: [0, 1, 2, 3],
    spectral: [4, 5, 6, 7, 8],
    timefreq: [9, 10, 11],
    constellation: [12, 13, 14, 15, 16],
    cyclostationary: [17, 18, 19, 20, 21, 22]
  };
  var FALLBACK_FEATURE_NAMES = [
    'cum_c40', 'cum_c41', 'cum_c42', 'cum_c63',
    'spec_centroid', 'spec_spread', 'spec_skewness', 'spec_kurtosis', 'spec_flatness',
    'tf_entropy', 'tf_peak_ratio', 'tf_ridge_slope',
    'const_eVM', 'const_mag_var', 'const_phase_var', 'const_radii', 'const_sym_count',
    'cyc_alpha_peak', 'cyc_peak_ratio', 'cyc_scd_entropy', 'cyc_coh_max', 'cyc_harmonic_1', 'cyc_harmonic_2'
  ];

  var COL = {
    grid: 'rgba(56, 72, 90, 0.40)',
    gridStrong: 'rgba(56, 72, 90, 0.75)',
    axisText: '#6e7e8d',
    axisText2: '#a4b2c0',
    line: '#2ed3b7',
    fill: 'rgba(46, 211, 183, 0.055)',
    floor: '#8494a4',
    band: 'rgba(46, 211, 183, 0.10)',
    bandEdge: 'rgba(46, 211, 183, 0.42)',
    true: '#e0a23a',
    resid: '#e0a23a',
    peak: '#dbe4ec',
    iq: 'rgba(46, 211, 183, 0.72)',
    frame: 'rgba(40, 53, 63, 0.9)',
    /* 解调页：同步轨迹三条曲线各用一色（编码而非装饰） */
    syncTiming: '#2ed3b7',
    syncPhase: '#e0a23a',
    syncFreq: '#57c98a',
    regionFill: 'rgba(164, 178, 192, 0.10)',
    regionLine: 'rgba(164, 178, 192, 0.45)'
  };
  var MONO_FONT = '11px ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace';
  var MONO_FONT_SM = '10px ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace';

  /* ================= 状态 ================= */
  var state = {
    mock: false,
    dir: 'data/demo',
    path: null,
    cases: [],
    filter: '',
    activeTab: 'spectrum',
    analyze: null,
    spectrum: null,
    waterfall: null,
    features: null,
    classify: null,
    constellation: null,
    demod: null,
    fec: null,
    frame: null,
    ssca: null,
    narrate: null,
    /* 设置页（LLM 配置，契约 §17）：config 是脱敏后的服务端结构，绝不包含密钥明文。
       baseline 是表单"上次已保存"的快照，用于只提交改动过的字段。
       apiKeyDraft 只存在于内存与 password 输入框，绝不写进页面文本/日志。 */
    llm: {
      config: null,
      loading: false,
      failed: false,
      error: '',
      models: [],
      modelsLoaded: false,
      modelsSig: '',
      modelsError: null,
      modelsSource: '',
      /* 服务商预设：来自 GET /api/llm/config 的 presets（可能缺失）；nodes 只缓存按钮节点，
         presetNotesId 记录最后一次点击，用于没有精确匹配时仍能展示 notes。 */
      presetsSig: '',
      presetNodes: [],
      presetNotesId: '',
      testing: false,
      test: null,
      testError: '',
      baseline: null,
      apiKeyDraft: '',
      clearPending: false,
      showKey: false,
      saveState: 'idle',
      saveMessage: '—'
    },
    /* 代理页（契约 §18）：tools 是 GET /api/agent/tools 的原始响应（budgets 在里面），
       run 是 POST /api/agent/run 的响应；gen 用于作废"路径已变"的在途运行结果。 */
    agent: {
      tools: null,
      budgets: null,
      toolsFailed: false,
      run: null,
      runPath: null,
      failed: false,
      error: '',
      running: false,
      startedAt: 0,
      elapsedMs: 0,
      timer: 0,
      gen: 0,
      updatedAt: ''
    },
    /* 访问令牌（契约 §19）：token 只存在内存 + localStorage + password 输入框，
       绝不进 textContent / title / 错误信息。persistFailed 表示浏览器拒绝写 localStorage。 */
    auth: { status: null, failed: false, error: '', token: '', persistFailed: false },
    pending: {},
    failed: {},
    errors: [],
    alerts: [],
    ui: {
      nperseg: 1024, nfft: 8192, points: 1200, scale: 'db', residual: true,
      demodMod: '', demodMaxSymbols: 2000, demodTracePoints: 400, demodRegions: true,
      fecMetric: 'ber', fecHidden: {},
      frameMod: '', frameSel: 0,
      sscaNfft: 256, sscaMaxPoints: 160,
      narrateProvider: 'auto',
      agentProvider: 'auto', agentRoundsManual: false
    },
    dataGen: 0,
    casesGen: 0,
    updatedAt: '',
    resizeRaf: 0,
    autoPicked: false
  };

  var el = {};
  var offCanvas = null;
  var COLORMAP = null;

  /* ================= 通用工具 ================= */
  function isNum(v) { return typeof v === 'number' && isFinite(v); }

  function $(id) { return document.getElementById(id); }

  function clear(node) { while (node && node.firstChild) { node.removeChild(node.firstChild); } }

  function show(node, on) {
    if (!node) { return; }
    if (on) {
      node.removeAttribute('hidden');
      node.style.display = '';
    } else {
      node.setAttribute('hidden', 'hidden');
      // 双保险：内联样式优先级高于任何样式表规则。
      // 曾经因为作者样式 .plot-empty{display:flex} 与 [hidden]{display:none} 同特异性、
      // 后者被覆盖，导致"数据已到、图也画了，加载覆盖层却一直盖在上面"。
      node.style.display = 'none';
    }
  }

  function mk(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (text !== undefined && text !== null) { n.textContent = String(text); }
    return n;
  }

  function setText(id, text) {
    var n = el[id];
    if (n) { n.textContent = (text === undefined || text === null) ? '—' : String(text); }
  }

  function setDim(id, on) {
    var n = el[id];
    if (!n) { return; }
    if (on) { if (n.className.indexOf('is-dim') < 0) { n.className += ' is-dim'; } }
    else { n.className = n.className.replace(' is-dim', ''); }
  }

  function hasKeys(o) {
    if (!o || typeof o !== 'object') { return false; }
    for (var k in o) { if (Object.prototype.hasOwnProperty.call(o, k)) { return true; } }
    return false;
  }

  function nowStamp() {
    var d = new Date();
    function p(x) { return (x < 10 ? '0' : '') + x; }
    return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
  }

  function delay(value, ms) {
    return new Promise(function (resolve) { setTimeout(function () { resolve(value); }, ms); });
  }

  /* ================= 数值格式化 ================= */
  function fmtEst(v) {
    if (!isNum(v)) { return '—'; }
    var a = Math.abs(v);
    if (a === 0) { return '0.00'; }
    if (a >= 1e6 || a < 1e-2) { return v.toExponential(2); }
    if (a >= 1e4) { return String(Math.round(v)); }
    if (a >= 1) { return v.toFixed(2); }
    return v.toFixed(3);
  }

  function fmtFeature(v) {
    if (!isNum(v)) { return '—'; }
    var a = Math.abs(v);
    if (v !== 0 && (a >= 1e5 || a < 1e-4)) { return v.toExponential(2); }
    return v.toFixed(4);
  }

  function fmtFixed(v, d) {
    if (!isNum(v)) { return '—'; }
    return v.toFixed(d);
  }

  function fmtInt(v) {
    if (!isNum(v)) { return '—'; }
    return String(Math.round(v));
  }

  function fmtCompact(v) {
    if (!isNum(v)) { return '—'; }
    var a = Math.abs(v);
    if (a >= 1e9) { return (v / 1e9).toFixed(3) + ' G'; }
    if (a >= 1e6) { return (v / 1e6).toFixed(3) + ' M'; }
    if (a >= 1e3) { return (v / 1e3).toFixed(2) + ' k'; }
    if (a >= 1) { return v.toFixed(2); }
    if (a === 0) { return '0'; }
    return v.toExponential(2);
  }

  function fmtHzTick(v) {
    var a = Math.abs(v);
    if (a >= 1e6) { return (v / 1e6).toFixed(2) + 'M'; }
    if (a >= 1e3) { return (v / 1e3).toFixed(1) + 'k'; }
    return String(Math.round(v * 100) / 100);
  }

  function fmtTimeTick(v) {
    var a = Math.abs(v);
    if (a === 0) { return '0'; }
    if (a < 1e-3) { return (v * 1e6).toFixed(0) + 'µs'; }
    if (a < 1) { return (v * 1e3).toFixed(a < 0.1 ? 1 : 0) + 'ms'; }
    return v.toFixed(a < 10 ? 2 : 1) + 's';
  }

  function fmtPercent(v, d) {
    if (!isNum(v)) { return '—'; }
    return (v * 100).toFixed(d === undefined ? 2 : d) + '%';
  }

  function confClass(c) {
    if (!isNum(c)) { return 'na'; }
    if (c >= 0.7) { return 'hi'; }
    if (c >= 0.4) { return 'mid'; }
    return 'lo';
  }

  function confText(c) {
    if (!isNum(c)) { return '置信度 —'; }
    var k = confClass(c);
    var t = k === 'hi' ? '高' : (k === 'mid' ? '中' : '低');
    return '置信度 ' + c.toFixed(2) + ' · ' + t;
  }

  function percentile(arr, q) {
    var tmp = [];
    var i;
    for (i = 0; i < arr.length; i++) { if (isNum(arr[i])) { tmp.push(arr[i]); } }
    if (!tmp.length) { return NaN; }
    tmp.sort(function (a, b) { return a - b; });
    var pos = (tmp.length - 1) * q;
    var lo = Math.floor(pos);
    var hi = Math.ceil(pos);
    if (lo === hi) { return tmp[lo]; }
    return tmp[lo] + (tmp[hi] - tmp[lo]) * (pos - lo);
  }

  function ticks(min, max, count) {
    var out = [];
    if (!isNum(min) || !isNum(max) || max <= min) { return out; }
    for (var i = 0; i <= count; i++) { out.push(min + (max - min) * i / count); }
    return out;
  }

  /* ================= JSON 高亮 ================= */
  function appendJsonHighlight(pre, text) {
    var re = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
    var last = 0;
    var m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) { pre.appendChild(document.createTextNode(text.slice(last, m.index))); }
      var cls = 'jnum';
      if (m[1] !== undefined) { cls = (m[2] !== undefined) ? 'jk' : 'jstr'; }
      else if (m[3] !== undefined) { cls = 'jbool'; }
      var span = mk('span', cls, m[0]);
      pre.appendChild(span);
      last = m.index + m[0].length;
      if (m[0].length === 0) { re.lastIndex++; }
    }
    if (last < text.length) { pre.appendChild(document.createTextNode(text.slice(last))); }
  }

  function renderJson(value) {
    var pre = mk('pre', 'json');
    var text;
    try { text = JSON.stringify(value, null, 2); } catch (err) { text = String(value); }
    if (text === undefined) { text = String(value); }
    appendJsonHighlight(pre, text);
    return pre;
  }

  function evidenceBlock(title, data) {
    var box = mk('details', 'evidence');
    box.appendChild(mk('summary', null, title));
    box.appendChild(renderJson(data));
    return box;
  }

  /* ================= 错误条 ================= */
  function recordError(api, status, message, url) {
    var key = api + '|' + status + '|' + message;
    var i;
    for (i = 0; i < state.errors.length; i++) {
      if (state.errors[i].key === key) {
        state.errors[i].count++;
        state.errors[i].time = nowStamp();
        renderErrors();
        return;
      }
    }
    state.errors.push({
      key: key, api: api, status: status,
      message: message, url: url || '', count: 1, time: nowStamp()
    });
    renderErrors();
  }

  function clearErrors() {
    state.errors = [];
    renderErrors();
  }

  /* 一键重试：把所有"卡住/失败"的请求重新发一遍。
     没有这个按钮时，只要有一发请求失败或超时，对应面板就会一直停在"加载中…/加载失败"，
     用户除了刷新页面没有别的办法 —— 这正是"图表一直在加载"的成因。 */
  function retryAll() {
    clearErrors();
    state.pending = {};
    state.failed = {};
    state.updatedAt = '';
    loadHealth();
    loadCases();
    if (state.activeTab === 'fec') {
      state.fec = null;
      loadFec();
    }
    if (state.path) {
      loadPath(state.path, { push: false });
    } else {
      ensureTabData(state.activeTab);
    }
    renderAll();
  }

  function renderErrors() {
    var host = el.errorList;
    if (!host) { return; }
    clear(host);
    var n = state.errors.length;
    if (el.errorCount) { el.errorCount.textContent = String(n) + (n > 1 ? ' 条' : ' 条'); }
    show(el.errorBar, n > 0);
    for (var i = 0; i < n; i++) {
      var e = state.errors[i];
      var li = mk('li', 'error-item' + (e.status === 401 ? ' is-auth' : ''));
      var api = mk('span', 'e-api', e.api);
      li.appendChild(api);
      var st = mk('span', 'e-status', e.status ? ('HTTP ' + e.status) : '网络');
      li.appendChild(st);
      li.appendChild(mk('span', 'e-msg', e.message));
      if (e.url) {
        var u = mk('span', 'e-url', e.url);
        u.title = e.url;
        li.appendChild(u);
      }
      /* 401 不当普通错误静默掉：给一个直达设置页的动作，并聚焦令牌输入框 */
      if (e.status === 401) {
        var go = mk('button', 'btn btn-xs e-auth-go', '去设置');
        go.type = 'button';
        go.addEventListener('click', function () {
          activateTab('settings');
          if (el.authToken) { try { el.authToken.focus(); } catch (ignore) { /* 焦点失败无妨 */ } }
        });
        li.appendChild(go);
      }
      if (e.count > 1) { li.appendChild(mk('span', 'e-status', 'x' + e.count)); }
      li.appendChild(mk('span', 'e-status', e.time));
      host.appendChild(li);
    }
    renderStatus();
  }

  /* ================= API 层 ================= */
  function buildQuery(params) {
    var parts = [];
    for (var k in params) {
      if (!Object.prototype.hasOwnProperty.call(params, k)) { continue; }
      var v = params[k];
      if (v === null || v === undefined || v === '') { continue; }
      parts.push(encodeURIComponent(k) + '=' + encodeURIComponent(String(v)));
    }
    return parts.join('&');
  }

  var MOCK_SUFFIXES = ['.sigmf-meta', '.sigmf-data', '.truth.npz', '.truth.json', '.npz', '.wav', '.raw', '.json'];

  function mockKeyFor(path) {
    if (!path) { return null; }
    var base = String(path).replace(/\\/g, '/');
    var seg = base.split('/');
    var name = seg[seg.length - 1];
    var i;
    for (i = 0; i < MOCK_SUFFIXES.length; i++) {
      var suf = MOCK_SUFFIXES[i];
      if (name.length > suf.length && name.slice(name.length - suf.length) === suf) {
        return name.slice(0, name.length - suf.length);
      }
    }
    return name;
  }

  function apiError(api, status, message, url) {
    var e = new Error(message);
    e.isApiError = true;
    e.api = api;
    e.status = status;
    e.url = url;
    return e;
  }

  function apiUrl(name, params, mockFile) {
    if (state.mock) {
      /* fec 与 cases/health 一样是全局端点，不按案例 slug 取文件（契约 §11 无查询参数）。
         mockFile：帧页的兜底文件名（<slug>.frame.json 不存在时改读 frame.json）。 */
      if (mockFile) { return MOCK_PREFIX + '/' + mockFile + '.json'; }
      if (name === 'cases' || name === 'health' || name === 'fec') { return MOCK_PREFIX + '/' + name + '.json'; }
      var key = mockKeyFor(params && params.path ? params.path : '');
      return MOCK_PREFIX + '/' + key + '.' + name + '.json';
    }
    var q = buildQuery(params || {});
    return API_PREFIX + '/' + name + (q ? ('?' + q) : '');
  }

  /* 单次请求超时：没有它，一个卡住的 fetch 会让面板永远停在"加载中…" */
  var API_TIMEOUT_MS = 25000;
  var API_SLOW_TIMEOUT_MS = { demod: 90000, frame: 90000, classify: 90000, features: 60000, constellation: 60000, waterfall: 60000, spectrum: 60000, ssca: 90000, narrate: 90000 };

  /* 去重：完全相同的（方法 + URL + body）请求在途时复用同一个 Promise。
     收益：标签页来回切、重试按钮连点都不会把同一端点打成两发。 */
  var INFLIGHT = {};

  /* 访问令牌（契约 §19）：localStorage 键名 + 401 统一文案。
     401 文案是固定串，绝不拼接响应体 —— 万一后端把 Authorization 回显出来，
     也不会顺着错误信息流进错误条 / 控制台。 */
  var AUTH_TOKEN_KEY = 'spxh.auth_token';
  var AUTH_401_MESSAGE = '需要访问令牌：请在设置页填入令牌';

  function apiLabelFor(url) {
    var u = String(url || '');
    var i = u.indexOf('/api/');
    if (i >= 0) { return u.slice(i + 5).split('?')[0]; }
    var seg = u.split('?')[0].split('/');
    return seg[seg.length - 1] || 'api';
  }

  function endsWithStr(s, suf) {
    return s.length >= suf.length && s.slice(s.length - suf.length) === suf;
  }

  function apiCore(method, url, body, timeoutMs, api) {
    var key = method + ' ' + url + ((body === null || body === undefined) ? '' : (' ' + JSON.stringify(body)));
    if (INFLIGHT[key]) { return INFLIGHT[key]; }
    var controller = (typeof AbortController !== 'undefined') ? new AbortController() : null;
    var timer = null;
    if (controller) { timer = setTimeout(function () { controller.abort(); }, timeoutMs); }
    var options = { method: method, headers: { 'Accept': 'application/json' }, cache: 'no-store' };
    if (body !== null && body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    /* 有令牌才带（留空则完全不带，与"默认不鉴权"的后端语义一致）；令牌只出现在请求头里 */
    if (state.auth && state.auth.token) { options.headers['Authorization'] = 'Bearer ' + state.auth.token; }
    if (controller) { options.signal = controller.signal; }
    var promise = fetch(url, options).then(function (res) {
      return res.text().then(function (text) {
        var parsed = null;
        if (text) { try { parsed = JSON.parse(text); } catch (err) { parsed = null; } }
        if (!res.ok) {
          /* 401 单独给可操作的中文提示（契约 §19）：不把响应体拼进来，避免敏感串随错误传播 */
          var msg = (res.status === 401) ? AUTH_401_MESSAGE
            : ((parsed && parsed.error) ? String(parsed.error)
              : (text ? text.replace(/\s/g, ' ').slice(0, 180) : ('HTTP ' + res.status)));
          throw apiError(api, res.status, msg, url);
        }
        if (parsed === null) { throw apiError(api, res.status, '响应不是合法 JSON', url); }
        return parsed;
      });
    }).then(function (parsed) {
      if (timer) { clearTimeout(timer); }
      delete INFLIGHT[key];
      return parsed;
    }, function (err) {
      if (timer) { clearTimeout(timer); }
      delete INFLIGHT[key];
      if (err && err.isApiError) { throw err; }
      var aborted = err && (err.name === 'AbortError' || String(err.message || '').toLowerCase().indexOf('abort') >= 0);
      if (aborted) {
        throw apiError(api, 0, '请求超时（' + Math.round(timeoutMs / 1000) + ' 秒无响应），点「重试全部」重发', url);
      }
      throw apiError(api, 0, '网络错误：无法连接后端（' + ((err && err.message) ? err.message : 'fetch failed') + '）', url);
    });
    INFLIGHT[key] = promise;
    return promise;
  }

  function apiGet(name, params, mockFile) {
    var url = apiUrl(name, params, mockFile);
    var timeoutMs = API_SLOW_TIMEOUT_MS[name] || API_TIMEOUT_MS;
    return apiCore('GET', url, null, timeoutMs, name).then(function (body) {
      return state.mock ? delay(body, 150) : body;
    });
  }

  /* POST：与 apiGet 共用同一套超时 / 错误条（由调用方 recordError）/ 去重策略。
     ?mock=1 时不真的发 POST，改为读对应假数据（配置类端点做最小合并），便于离线演示。 */
  function apiPost(url, body, opts) {
    opts = opts || {};
    var api = opts.api || apiLabelFor(url);
    var timeoutMs = opts.timeoutMs || API_TIMEOUT_MS;
    var payload = (body && typeof body === 'object') ? body : {};
    if (state.mock) { return mockPost(url, payload, api); }
    return apiCore('POST', url, payload, timeoutMs, api);
  }

  function mockFileForPost(url) {
    if (endsWithStr(url, '/llm/config')) { return 'llm_config'; }
    if (endsWithStr(url, '/llm/test')) { return 'llm_test'; }
    if (endsWithStr(url, '/agent/run')) { return 'agent_run'; }
    return null;
  }

  function mockGetJson(file, api) {
    return apiCore('GET', MOCK_PREFIX + '/' + file + '.json', null, API_TIMEOUT_MS, api);
  }

  function mockPost(url, body, api) {
    var mf = mockFileForPost(url);
    if (!mf) { return Promise.reject(apiError(api, 404, 'mock 模式不支持该 POST 端点', url)); }
    /* 代理运行：先读 agent_run.json（LLM 结果），404 再回退 agent_run_offline.json（离线确定性计划），
       与契约 §18 "没有可用 LLM 时走离线确定性代理" 的两种形态都能离线演示。 */
    if (mf === 'agent_run') {
      return mockGetJson('agent_run', api).then(function (base) {
        return delay(base, 150);
      }, function (err) {
        if (err && err.status === 404) {
          return mockGetJson('agent_run_offline', api).then(function (base) { return delay(base, 150); });
        }
        throw err;
      });
    }
    return mockGetJson(mf, api).then(function (base) {
      var out = base;
      if (mf === 'llm_config') { out = mergeMockLlmConfig(base, body); }
      return delay(out, 150);
    });
  }

  /* 与后端 normalize_base_url 对齐的离线模拟（仅 mock 模式用）：
     让"粘了 .../chat/completions"在离线演示里也能被纠正成根地址，行为与真机一致。 */
  function mockNormalizeBaseUrl(raw) {
    var text = String(raw === null || raw === undefined ? '' : raw).replace(/\s+$/, '');
    while (text.length && text.charAt(text.length - 1) === '/') { text = text.slice(0, -1); }
    var suffixes = ['/chat/completions', '/completions', '/models'];
    for (var i = 0; i < suffixes.length; i++) {
      var suf = suffixes[i];
      if (text.length >= suf.length && text.slice(text.length - suf.length) === suf) {
        text = text.slice(0, text.length - suf.length);
        while (text.length && text.charAt(text.length - 1) === '/') { text = text.slice(0, -1); }
      }
    }
    return text;
  }

  /* mock 保存：把提交字段合并进假配置，并同步脱敏字段（api_key 只留尾 4 位）。 */
  function mergeMockLlmConfig(base, body) {
    var cfg = (base && typeof base === 'object') ? base : {};
    var out = {};
    var k;
    for (k in cfg) { if (Object.prototype.hasOwnProperty.call(cfg, k)) { out[k] = cfg[k]; } }
    var editable = ['enabled', 'base_url', 'model', 'temperature', 'max_tokens', 'timeout_s'];
    for (var i = 0; i < editable.length; i++) {
      var f = editable[i];
      if (body[f] === undefined) { continue; }
      out[f] = (f === 'base_url' && typeof body[f] === 'string') ? mockNormalizeBaseUrl(body[f]) : body[f];
      if (!out.source || typeof out.source !== 'object') { out.source = {}; }
      out.source[f] = 'file';
    }
    if (body.clear_api_key === true) {
      out.api_key_set = false;
      out.api_key_hint = null;
      if (!out.source || typeof out.source !== 'object') { out.source = {}; }
      out.source.api_key = 'file';
    } else if (typeof body.api_key === 'string' && body.api_key) {
      out.api_key_set = true;
      out.api_key_hint = maskKeyHint(body.api_key);
      if (!out.source || typeof out.source !== 'object') { out.source = {}; }
      out.source.api_key = 'file';
    }
    out.ready = (out.enabled !== false) && (out.api_key_set === true);
    return out;
  }

  /* 只回显尾 4 位：前端本地也守同样的规则，避免 mock / 输入回显泄露完整密钥。 */
  function maskKeyHint(key) {
    var s = String(key || '');
    if (!s) { return null; }
    return s.length <= 4 ? '****' : ('****' + s.slice(s.length - 4));
  }

  function loadDataset(key, name, params, apply) {
    var gen = state.dataGen;
    state.pending[key] = true;
    state.failed[key] = false;
    renderAll();
    apiGet(name, params).then(function (data) {
      if (gen !== state.dataGen) { return; }
      state.pending[key] = false;
      state.failed[key] = false;
      state.updatedAt = nowStamp();
      apply(data);
      renderAll();
    }, function (err) {
      if (gen !== state.dataGen) { return; }
      state.pending[key] = false;
      state.failed[key] = true;
      recordError(name, err.status, err.message, err.url);
      renderAll();
    });
  }

  /* ================= 数据加载 ================= */
  function loadHealth() {
    apiGet('health', {}).then(function (d) {
      if (el.healthEstimator) {
        el.healthEstimator.textContent = '估计器 ' + (d && d.estimator ? d.estimator : '—');
        el.healthEstimator.title = 'version ' + (d && d.version ? d.version : '—');
        el.healthEstimator.className = 'meta-chip is-on';
      }
      if (el.healthClassifier) {
        if (d && d.classifier) {
          el.healthClassifier.textContent = '分类器 ' + d.classifier;
          el.healthClassifier.className = 'meta-chip is-on';
        } else {
          el.healthClassifier.textContent = '分类器 未训练';
          el.healthClassifier.className = 'meta-chip is-warn';
        }
      }
    }, function (err) {
      recordError('health', err.status, err.message, err.url);
      if (el.healthEstimator) { el.healthEstimator.textContent = '估计器 不可用'; el.healthEstimator.className = 'meta-chip is-off'; }
      if (el.healthClassifier) { el.healthClassifier.textContent = '分类器 不可用'; el.healthClassifier.className = 'meta-chip is-off'; }
    });
  }

  function loadCases() {
    var gen = ++state.casesGen;
    state.pending.cases = true;
    renderCases();
    renderStatus();
    apiGet('cases', { dir: state.dir }).then(function (d) {
      if (gen !== state.casesGen) { return; }
      state.pending.cases = false;
      state.failed.cases = false;
      state.cases = (d && Array.isArray(d.cases)) ? d.cases : [];
      if (d && d.dir) { state.dir = String(d.dir); if (el.dirInput) { el.dirInput.value = state.dir; } }
      renderCases();
      renderStatus();
      maybeAutoPick();
    }, function (err) {
      if (gen !== state.casesGen) { return; }
      state.pending.cases = false;
      state.failed.cases = true;
      state.cases = [];
      recordError('cases', err.status, err.message, err.url);
      renderCases();
      renderStatus();
    });
  }

  function maybeAutoPick() {
    if (state.autoPicked || state.path || !state.cases.length) { return; }
    state.autoPicked = true;
    var p = caseDataPath(state.cases[0]);
    if (p) { loadPath(p, { push: false }); }
  }

  function caseDataPath(c) {
    if (!c) { return null; }
    var f = c.files || {};
    var order = ['sigmf_meta', 'sigmf_data', 'npz', 'wav', 'raw', 'truth_npz'];
    for (var i = 0; i < order.length; i++) {
      if (f[order[i]]) { return String(f[order[i]]); }
    }
    return c.prefix ? String(c.prefix) : null;
  }

  function loadPath(path, opts) {
    opts = opts || {};
    state.path = path;
    state.dataGen++;
    state.analyze = null;
    state.spectrum = null;
    state.waterfall = null;
    state.features = null;
    state.classify = null;
    state.constellation = null;
    state.demod = null;
    state.frame = null;
    state.ssca = null;
    state.narrate = null;
    /* 代理结果与路径强绑定：换路径就作废（含在途运行），避免把上一个案例的结论留在页面上 */
    state.agent.gen++;
    state.agent.run = null;
    state.agent.runPath = null;
    state.agent.failed = false;
    state.agent.error = '';
    state.agent.elapsedMs = 0;
    state.agent.running = false;
    agentStopTicker();
    sscaHover = null;
    /* /api/fec 与路径无关：保留在途请求状态，避免自动选址时对同一份报告重复请求 */
    var fecPending = !!state.pending.fec;
    state.pending = {};
    state.failed = {};
    if (fecPending) { state.pending.fec = true; }
    state.updatedAt = '';
    clearErrors();
    if (el.pathInput) { el.pathInput.value = path || ''; }
    syncUrl(opts.push !== false);
    renderCases();
    renderAll();
    loadCore();
    ensureTabData(state.activeTab);
  }

  function loadCore() {
    if (!state.path) { return; }
    loadDataset('analyze', 'analyze',
      { path: state.path, nperseg: state.ui.nperseg, nfft: state.ui.nfft || null },
      function (d) { state.analyze = d; });
    loadSpectrum();
    loadWaterfall();
  }

  function loadSpectrum() {
    if (!state.path) { return; }
    loadDataset('spectrum', 'spectrum',
      { path: state.path, nperseg: state.ui.nperseg, nfft: state.ui.nfft || null, points: state.ui.points },
      function (d) { state.spectrum = d; });
  }

  function loadWaterfall() {
    if (!state.path) { return; }
    var np = state.ui.nperseg > 512 ? 512 : state.ui.nperseg;
    loadDataset('waterfall', 'waterfall',
      { path: state.path, nperseg: np, max_frames: 400, max_bins: 192 },
      function (d) { state.waterfall = d; });
  }

  function ensureTabData(tab) {
    /* /api/fec 是全局报告，不依赖当前案例路径；已有数据或请求在途都不再重复请求 */
    if (tab === 'fec') {
      if (!state.fec && !state.pending.fec && !state.failed.fec) { loadFec(); }
      return;
    }
    if (tab === 'frame') {
      if (!state.frame && !state.pending.frame && !state.failed.frame) { loadFrame(); }
      return;
    }
    if (tab === 'ssca') {
      /* 已有数据 / 请求在途 / 已失败（等用户点「重新计算」）都不重复请求 */
      if (!state.ssca && !state.pending.ssca && !state.failed.ssca) { loadSsca(); }
      return;
    }
    if (tab === 'narrate') {
      /* 报告依赖当前路径：已有数据 / 请求在途 / 已失败（等用户点「重新生成」）都不重复请求 */
      if (!state.narrate && !state.pending.narrate && !state.failed.narrate) { loadNarrate(); }
      /* 报告页要显示 LLM 就绪提示：只发一次 GET 配置（与设置页共用缓存，切标签不重复请求） */
      ensureLlmConfig();
      return;
    }
    if (tab === 'agent') {
      /* 进入代理页只拉一次预算/工具清单（GET）。绝不自动运行：POST /api/agent/run 花钱且慢，
         必须由用户点「运行代理」触发（本条是任务里的硬要求）。 */
      ensureAgentTools();
      return;
    }
    if (tab === 'settings') {
      /* 设置页只发 GET；首次进入拉一次配置 + 一次鉴权状态，之后切回来不再请求
         （失败时分别由「刷新配置」「刷新状态」显式重试） */
      ensureLlmConfig();
      ensureAuthStatus();
      return;
    }
    if (!state.path) { return; }
    if (tab === 'features') {
      if (!state.features && !state.pending.features) {
        loadDataset('features', 'features', { path: state.path },
          function (d) { state.features = d; });
      }
      if (!state.constellation && !state.pending.constellation) {
        loadDataset('constellation', 'constellation', { path: state.path, max_points: 3000 },
          function (d) { state.constellation = d; });
      }
    } else if (tab === 'classify') {
      if (!state.classify && !state.pending.classify) {
        loadDataset('classify', 'classify', { path: state.path },
          function (d) { state.classify = d; });
      }
    } else if (tab === 'demod') {
      /* 已有数据或请求在途都不再发起第二个请求；失败后由「重解调」按钮显式重试 */
      if (!state.demod && !state.pending.demod && !state.failed.demod) {
        loadDemod();
      }
    }
  }

  function loadDemod() {
    if (!state.path) { return; }
    var params = {
      path: state.path,
      max_symbols: state.ui.demodMaxSymbols,
      trace_points: state.ui.demodTracePoints
    };
    if (state.ui.demodMod) { params.modulation = state.ui.demodMod; }
    loadDataset('demod', 'demod', params, function (d) { state.demod = d; });
  }

  /* /api/fec 无查询参数（契约 §11，数据由 check_m4 预计算落盘）。
     它是全局报告、与案例路径无关，所以用独立代际计数器守护：
     切换案例不会作废在途请求，也不会对同一份报告发第二次请求。 */
  var fecGen = 0;

  function loadFec() {
    var gen = ++fecGen;
    state.pending.fec = true;
    state.failed.fec = false;
    renderAll();
    apiGet('fec', {}).then(function (d) {
      if (gen !== fecGen) { return; }
      state.pending.fec = false;
      state.failed.fec = false;
      state.updatedAt = nowStamp();
      state.fec = d;
      renderAll();
    }, function (err) {
      if (gen !== fecGen) { return; }
      state.pending.fec = false;
      state.failed.fec = true;
      recordError('fec', err.status, err.message, err.url);
      renderAll();
    });
  }

  /* ---------- M5 帧同步与载荷提取 ---------- */
  /* /api/frame 依赖当前路径（与 demod 同类），但需要 mock 兜底：
     ?mock=1 先读 <slug>.frame.json，404 再读全局 frame.json；失败/超时都落到 failed 态。 */
  function loadFrame(mockFile) {
    if (!state.path) { return; }
    var gen = state.dataGen;
    state.pending.frame = true;
    state.failed.frame = false;
    state.ui.frameSel = 0;
    renderAll();
    var params = { path: state.path };
    if (state.ui.frameMod) { params.modulation = state.ui.frameMod; }
    apiGet('frame', params, mockFile).then(function (d) {
      if (gen !== state.dataGen) { return; }
      state.pending.frame = false;
      state.failed.frame = false;
      state.updatedAt = nowStamp();
      state.frame = d;
      renderAll();
    }, function (err) {
      if (gen !== state.dataGen) { return; }
      if (state.mock && !mockFile && err.status === 404) { loadFrame('frame'); return; }
      state.pending.frame = false;
      state.failed.frame = true;
      recordError('frame', err.status, err.message, err.url);
      renderAll();
    });
  }

  function frameData() { return state.frame; }

  function frameEmptyText(extra) {
    if (state.failed.frame) { return extra ? ('帧数据加载失败（见顶部错误条）：' + extra) : '帧数据加载失败（见顶部错误条）'; }
    if (state.pending.frame) { return '加载中…'; }
    if (!state.path) { return '等待数据：请在左侧选择案例或输入路径'; }
    return extra ? ('暂无帧数据：' + extra) : '暂无帧数据（记录太短或未检出前导）';
  }

  function renderFrame() {
    renderFrameSync();
    renderFrameBlind();
    renderFrameCards();
    renderFrameTable();
    renderFramePayload();
    renderFrameWarnings();
    renderFrameCorrLegend();
  }

  function renderFrameCorrLegend() {
    var host = el.frameCorrLegend;
    if (!host) { return; }
    clear(host);
    if (!frameData()) { return; }
    host.appendChild(legendItem('legend-swatch', '相关幅度'));
    host.appendChild(legendItem('legend-swatch is-true', '帧起点'));
    host.appendChild(legendItem('legend-swatch is-dash', 'correlation_peak'));
  }

  function renderFrameSync() {
    var d = frameData();
    var sync = (d && d.sync && typeof d.sync === 'object') ? d.sync : null;
    var locked = !!(sync && sync.locked);
    if (el.fsLock) {
      if (!d) {
        el.fsLock.textContent = state.pending.frame ? '同步 加载中…'
          : (state.failed.frame ? '同步 加载失败' : '同步 —');
        el.fsLock.className = 'badge ' + (state.failed.frame ? 'badge-lo' : 'badge-na');
      } else if (!sync) {
        el.fsLock.textContent = '同步 无 sync 字段';
        el.fsLock.className = 'badge badge-na';
      } else {
        el.fsLock.textContent = locked ? '同步 已锁定' : '同步 未锁定';
        el.fsLock.className = 'badge ' + (locked ? 'badge-hi' : 'badge-lo');
      }
    }
    setText('fsPreamble', sync && sync.preamble ? String(sync.preamble) : '—');
    setText('fsPreambleBits', sync && sync.preamble_bits ? String(sync.preamble_bits) : '—');
    setText('fsMod', d && d.modulation ? String(d.modulation) : '—');
    setText('fsSrc', d && d.modulation_source ? ('来源 ' + String(d.modulation_source)) : '来源 —');
    setText('fsPeak', sync && isNum(sync.correlation_peak) ? sync.correlation_peak.toFixed(3) : '—');
    setText('fsPsl', sync && isNum(sync.peak_to_sidelobe_db) ? (sync.peak_to_sidelobe_db.toFixed(1) + ' dB') : '—');
    setText('fsSymbol', sync && isNum(sync.frame_start_symbol) ? fmtFixed(sync.frame_start_symbol, 1) : '—');
    setText('fsSample', sync && isNum(sync.frame_start_sample) ? fmtInt(sync.frame_start_sample) : '—');
    var deg = (sync && isNum(sync.ambiguity_rotation_deg)) ? sync.ambiguity_rotation_deg : null;
    setText('rotDeg', deg === null ? '—' : (signed(deg, 1) + '°'));
    if (el.rotNeedle) {
      el.rotNeedle.style.transform = deg === null ? 'rotate(0deg)' : ('rotate(' + String(deg) + 'deg)');
    }
    setText('rotQuad', deg === null ? '—' : (deg === 0 ? '无模糊' : '对称旋转 ' + signed(deg, 0) + '°'));
    if (el.rotNote) {
      el.rotNote.textContent = 'ambiguity_rotation_deg · Barker-13 相关解出，取代真值搜索';
    }
  }

  /* 盲恢复对象：契约 §15 规定放在 sync.blind。
     也兼容后端把同名字段放在顶层 blind 的情形；顶层 blind 也可能是布尔开关，布尔不算对象、直接忽略。 */
  function frameBlind(d) {
    var sync = (d && d.sync && typeof d.sync === 'object') ? d.sync : null;
    var b = (sync && sync.blind && typeof sync.blind === 'object') ? sync.blind : null;
    if (!b && d && d.blind && typeof d.blind === 'object') { b = d.blind; }
    return b;
  }

  /* 帧头 CRC-8 三态：true / false / null。null = 字段缺失（旧格式），按未知显示，绝不当失败。
     契约 §15 把 header_crc_ok / header_crc8 / crc8_expected 放在 frame.header 里；
     当前后端 extract.py 把它们放在帧顶层（frames[].header_crc_ok）。两边都读，优先契约位置。 */
  function frameHeaderCrc(header, frame) {
    if (header && typeof header.header_crc_ok === 'boolean') { return header.header_crc_ok; }
    if (frame && typeof frame.header_crc_ok === 'boolean') { return frame.header_crc_ok; }
    return null;
  }
  function headerCrcField(header, frame, key) {
    if (header && header[key] !== undefined && header[key] !== null && header[key] !== '') { return header[key]; }
    if (frame && frame[key] !== undefined && frame[key] !== null && frame[key] !== '') { return frame[key]; }
    return null;
  }

  function renderFrameBlind() {
    var box = el.frameBlind;
    if (!box) { return; }
    var b = frameBlind(frameData());
    show(box, !!b);
    if (!b) { return; }
    var hc = (typeof b.header_crc_ok === 'boolean') ? b.header_crc_ok : null;
    if (el.fbHeaderCrc) {
      el.fbHeaderCrc.textContent = hc === null ? '帧头 CRC —' : ('帧头 CRC ' + (hc ? '通过' : '失败'));
      el.fbHeaderCrc.className = 'badge ' + (hc === null ? 'badge-na' : (hc ? 'badge-hi' : 'badge-lo'));
    }
    var lr = (typeof b.length_recovered === 'boolean') ? b.length_recovered : null;
    if (el.fbLength) {
      el.fbLength.textContent = lr === null ? '长度恢复 —' : ('长度恢复 ' + (lr ? '成功' : '失败'));
      el.fbLength.className = 'badge ' + (lr === null ? 'badge-na' : (lr ? 'badge-hi' : 'badge-lo'));
    }
    setText('fbHypotheses', isNum(b.hypotheses_tried) ? fmtInt(b.hypotheses_tried) : '—');
    var rh = (b.repaired_header && typeof b.repaired_header === 'object') ? b.repaired_header : null;
    setText('fbPayloadLen', (rh && isNum(rh.payload_len)) ? (fmtInt(rh.payload_len) + ' B') : '—');
    setText('fbFec', (rh && rh.fec) ? String(rh.fec) : '—');
    var il = (rh && typeof rh.interleave === 'boolean') ? rh.interleave : null;
    setText('fbInterleave', il === null ? '—' : (il ? 'true' : 'false'));
  }

  function renderFrameCards() {
    var host = el.frameCards;
    if (!host) { return; }
    clear(host);
    var d = frameData();
    if (!d || !d.summary) { host.appendChild(mk('div', 'empty', frameEmptyText())); return; }
    var s = d.summary;
    var defs = [
      { title: '解析帧数', num: isNum(s.frames) ? fmtInt(s.frames) : '—', unit: '帧',
        method: 'summary.frames · 连续解析，CRC 失败帧也计入' },
      { title: 'CRC 通过', num: isNum(s.frames_crc_ok) ? fmtInt(s.frames_crc_ok) : '—',
        unit: isNum(s.frames) ? ('/ ' + fmtInt(s.frames)) : '',
        method: 'summary.frames_crc_ok · CRC-16-CCITT 覆盖帧头 + 载荷' },
      { title: '载荷字节', num: isNum(s.payload_bytes_total) ? fmtInt(s.payload_bytes_total) : '—', unit: 'B',
        method: 'summary.payload_bytes_total' },
      { title: '与真值逐字节一致', num: isNum(s.payload_bytes_exact) ? fmtInt(s.payload_bytes_exact) : '（无真值）',
        unit: isNum(s.payload_bytes_exact) ? 'B' : '',
        method: 'summary.payload_bytes_exact · 只在有 truth sidecar 时给出' }
    ];
    var i;
    for (i = 0; i < defs.length; i++) {
      var def = defs[i];
      var card = mk('div', 'est-card');
      var head = mk('div', 'est-head');
      head.appendChild(mk('span', 'est-title', def.title));
      card.appendChild(head);
      var line = mk('div', 'est-value');
      var numEl = mk('span', 'est-num', String(def.num));
      if (String(def.num) === '—') { numEl.className += ' is-na'; }
      line.appendChild(numEl);
      if (def.unit) { line.appendChild(mk('span', 'est-unit', String(def.unit))); }
      card.appendChild(line);
      card.appendChild(mk('div', 'est-method', String(def.method)));
      host.appendChild(card);
    }
  }

  function frameFlags(frame) {
    var crc = (frame && frame.crc && typeof frame.crc === 'object') ? frame.crc : {};
    return {
      preambleOk: !!(frame && frame.preamble_ok),
      crcOk: !!crc.ok
    };
  }

  function renderFrameTable() {
    var host = el.frameTable;
    if (!host) { return; }
    clear(host);
    var d = frameData();
    var frames = (d && Array.isArray(d.frames)) ? d.frames : null;
    setText('frameCount', frames ? (fmtInt(frames.length) + ' 帧') : '—');
    if (!frames || !frames.length) {
      host.appendChild(mk('div', 'empty', frameEmptyText()));
      return;
    }
    var table = mk('table', 'fec-table');
    var thead = mk('thead');
    var hr = mk('tr');
    var cols = ['#', '起始符号', '起始比特', '前导', '载荷', '帧头 CRC', 'CRC-8', 'CRC', '与真值误码数'];
    var i;
    for (i = 0; i < cols.length; i++) { hr.appendChild(mk('th', null, cols[i])); }
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = mk('tbody');
    for (i = 0; i < frames.length; i++) {
      var f = frames[i] || {};
      var flags = frameFlags(f);
      var header = (f.header && typeof f.header === 'object') ? f.header : {};
      var tr = mk('tr', i === state.ui.frameSel ? 'is-sel' : '');
      tr.setAttribute('data-frame-index', String(i));
      tr.appendChild(mk('td', 'mono', String(isNum(f.index) ? f.index : i)));
      tr.appendChild(mk('td', 'mono', isNum(f.start_symbol) ? fmtFixed(f.start_symbol, 1) : '—'));
      tr.appendChild(mk('td', 'mono', isNum(f.start_bit) ? fmtInt(f.start_bit) : '—'));
      tr.appendChild(mk('td', flags.preambleOk ? 'check-ok' : 'check-bad', flags.preambleOk ? '✓' : '✗'));
      tr.appendChild(mk('td', 'mono', isNum(header.payload_len) ? (fmtInt(header.payload_len) + ' B') : '—'));
      /* 帧头 CRC-8（契约 §15）：true=绿勾 false=红叉；字段缺失 = 旧格式，按未知显示，绝不当失败 */
      var hcrc = frameHeaderCrc(header, f);
      var hcrcCell = (hcrc === null)
        ? mk('td', 'check-na', '—（旧格式）')
        : mk('td', hcrc ? 'check-ok' : 'check-bad', hcrc ? '✓' : '✗');
      var got8raw = headerCrcField(header, f, 'header_crc8');
      var exp8raw = headerCrcField(header, f, 'crc8_expected');
      var got8 = (got8raw === null) ? null : String(got8raw);
      var exp8 = (exp8raw === null) ? null : String(exp8raw);
      if (got8 || exp8) {
        hcrcCell.title = 'header_crc8 ' + (got8 || '—') + ' · crc8_expected ' + (exp8 || '—');
      }
      tr.appendChild(hcrcCell);
      var crc8Cell = mk('td', 'mono', got8 ? String(got8) : '—');
      if (exp8) { crc8Cell.title = 'crc8_expected ' + String(exp8); }
      tr.appendChild(crc8Cell);
      var crcCell = mk('td', flags.crcOk ? 'check-ok' : 'check-bad', flags.crcOk ? '✓' : '✗');
      var crcObj = (f.crc && typeof f.crc === 'object') ? f.crc : {};
      if (crcObj.expected || crcObj.computed) {
        crcCell.title = 'expected ' + String(crcObj.expected) + ' · computed ' + String(crcObj.computed);
      }
      tr.appendChild(crcCell);
      tr.appendChild(mk('td', 'mono', isNum(f.bit_errors_vs_truth) ? String(f.bit_errors_vs_truth) : '无真值'));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    host.appendChild(table);
  }

  function renderFramePayload() {
    var host = el.payloadBody;
    if (!host) { return; }
    clear(host);
    var d = frameData();
    var frames = (d && Array.isArray(d.frames)) ? d.frames : null;
    if (!frames || !frames.length) {
      setText('payloadMeta', '—');
      host.appendChild(mk('div', 'empty', frameEmptyText()));
      return;
    }
    var sel = Math.min(Math.max(0, state.ui.frameSel | 0), frames.length - 1);
    var f = frames[sel] || {};
    var header = (f.header && typeof f.header === 'object') ? f.header : {};
    var pl = (f.payload && typeof f.payload === 'object') ? f.payload : {};
    if (el.payloadMeta) { el.payloadMeta.textContent = '第 ' + String(sel) + ' 帧 · ' + (isNum(pl.bytes) ? fmtInt(pl.bytes) : '—') + ' 字节'; }
    if (el.frameSelChip) { el.frameSelChip.textContent = '已选第 ' + String(sel) + ' 帧'; }
    var meta = mk('dl', 'kv');
    var hcrc = frameHeaderCrc(header, f);
    var hcrc8raw = headerCrcField(header, f, 'header_crc8');
    var ecrc8raw = headerCrcField(header, f, 'crc8_expected');
    var hcrc8 = (hcrc8raw === null) ? '—' : String(hcrc8raw);
    var ecrc8 = (ecrc8raw === null) ? '—' : String(ecrc8raw);
    var rows = [
      ['payload_len', isNum(header.payload_len) ? String(header.payload_len) : '—'],
      ['modulation', header.modulation ? String(header.modulation) : '—'],
      ['fec', header.fec ? String(header.fec) : '—'],
      ['interleave', (typeof header.interleave === 'boolean') ? (header.interleave ? 'true' : 'false') : '—'],
      ['header_crc_ok', hcrc === null ? '—（旧格式，字段缺失）' : (hcrc ? 'true（通过）' : 'false（失败）')],
      ['header_crc8 / crc8_expected', hcrc8 + ' / ' + ecrc8],
      ['header_hex', header.header_hex ? String(header.header_hex) : '—'],
      ['crc', (f.crc && f.crc.expected ? (String(f.crc.expected) + ' / ' + String(f.crc.computed) + ' ' + (f.crc.ok ? '(OK)' : '(FAIL)')) : '—')]
    ];
    var i;
    for (i = 0; i < rows.length; i++) {
      meta.appendChild(mk('dt', null, rows[i][0]));
      meta.appendChild(mk('dd', 'mono', rows[i][1]));
    }
    host.appendChild(meta);

    /* 逐字节预览：hex 优先；缺失时用 head（前 32 字节）拼；ascii 缺失时从字节自行生成。
       任何一个字段缺失都不影响另一段的展示。 */
    var hex = (typeof pl.hex === 'string') ? pl.hex.replace(/[^0-9a-fA-F]/g, '') : '';
    var i;
    if (!hex && Array.isArray(pl.head)) {
      var hp = [];
      for (i = 0; i < pl.head.length; i++) { hp.push(frameByteHex(pl.head[i])); }
      hex = hp.join('');
    }
    if (hex) {
      host.appendChild(mk('div', 'ber-sec-k', '载荷 HEX（' + String(Math.ceil(hex.length / 2)) + ' 字节' + (pl.truncated ? '，已截断' : '') + '）'));
      host.appendChild(mk('pre', 'bits-preview mono', groupHex(hex)));
    } else {
      host.appendChild(mk('div', 'empty', '该帧没有 payload.hex 也没有 payload.head'));
    }
    var ascii = (typeof pl.ascii === 'string') ? pl.ascii : '';
    if (!ascii && hex) {
      var chars = [];
      for (i = 0; i + 1 < hex.length; i += 2) {
        var code = parseInt(hex.substr(i, 2), 16);
        chars.push((code >= 0x20 && code <= 0x7e) ? String.fromCharCode(code) : '.');
      }
      ascii = chars.join('');
    }
    if (ascii) {
      host.appendChild(mk('div', 'ber-sec-k', '载荷 ASCII（不可打印字符显示为 .）'));
      host.appendChild(mk('pre', 'bits-preview mono', ascii));
    }
  }

  function frameByteHex(b) {
    var v = isNum(b) ? (b & 0xff) : 0;
    var s = v.toString(16).toUpperCase();
    return s.length < 2 ? ('0' + s) : s;
  }

  function groupHex(hex) {
    var out = [];
    var i;
    for (i = 0; i < hex.length; i += 2) { out.push(hex.substr(i, 2)); }
    var lines = [];
    for (i = 0; i < out.length; i += 16) { lines.push(out.slice(i, i + 16).join(' ')); }
    return lines.join('\n');
  }

  function renderFrameWarnings() {
    var host = el.frameWarnings;
    if (!host) { return; }
    clear(host);
    var d = frameData();
    if (!d) {
      setText('frameWarnCount', '—');
      host.appendChild(mk('div', 'empty', frameEmptyText()));
      return;
    }
    var list = Array.isArray(d.warnings) ? d.warnings : [];
    setText('frameWarnCount', fmtInt(list.length) + ' 条');
    if (!list.length) { host.appendChild(mk('div', 'empty', '无告警')); return; }
    var ul = mk('ul', 'warnlist');
    var i;
    for (i = 0; i < list.length; i++) { ul.appendChild(mk('li', null, String(list[i]))); }
    host.appendChild(ul);
  }

  /* 相关峰图：滑动相关幅度 vs 符号序号，标注每帧起点（竖线 + 圆点）与检测到的首帧起点 */
  function drawFrameCorr() {
    var canvas = el.frameCorrCanvas;
    if (!canvas) { return; }
    var emp = el.frameCorrEmpty;
    var d = frameData();
    var sync = (d && d.sync && typeof d.sync === 'object') ? d.sync : null;
    var corr = (sync && sync.correlation && typeof sync.correlation === 'object') ? sync.correlation : null;
    var idx = (corr && Array.isArray(corr.index)) ? corr.index : [];
    var val = (corr && Array.isArray(corr.value)) ? corr.value : [];
    var lim = Math.min(idx.length, val.length);
    /* 逐点容错：null/NaN 只跳过该点，不让整页崩 */
    var pts = [];
    var i;
    for (i = 0; i < lim; i++) {
      if (!isNum(idx[i]) || !isNum(val[i])) { continue; }
      pts.push({ x: idx[i], y: val[i] });
    }
    if (emp) {
      emp.textContent = !d ? frameEmptyText() : '暂无相关峰数据（响应中 sync.correlation 为空）';
      show(emp, pts.length === 0);
    }
    if (el.frameCorrMeta) {
      el.frameCorrMeta.textContent = pts.length ? ('有效 ' + fmtInt(pts.length) + ' 点') : '—';
    }
    if (!pts.length) { return; }

    var f = fitCanvas(canvas);
    var ctx = f.ctx;
    var m = { l: 52, r: 14, t: 12, b: 26 };
    var pw = f.w - m.l - m.r;
    var ph = f.h - m.t - m.b;
    if (pw < 24 || ph < 24) { return; }

    var xmin = pts[0].x;
    var xmax = pts[0].x;
    var ymax = 0;
    for (i = 0; i < pts.length; i++) {
      if (pts[i].x < xmin) { xmin = pts[i].x; }
      if (pts[i].x > xmax) { xmax = pts[i].x; }
      if (pts[i].y > ymax) { ymax = pts[i].y; }
    }
    if (xmax <= xmin) { xmax = xmin + 1; }
    if (ymax <= 0) { ymax = 1; }
    ymax = ymax * 1.10;

    /* 交互层：X = 符号序号，Y = 归一化相关幅度 */
    var win = chartWindow('frameCorr', { x0: xmin, x1: xmax, y0: 0, y1: ymax }, {
      xCount: pts.length, yCount: 0,
      fmtX: fmtInt, unitX: '符号',
      fmtY: function (v) { return v.toFixed(2); }, unitY: '',
      canvas: canvas, redraw: drawFrameCorr
    });
    xmin = win.x0; xmax = win.x1; ymax = win.y1;
    chartGeom('frameCorr', m, pw, ph);

    function xOf(v) { return m.l + (v - xmin) / (xmax - xmin) * pw; }
    function yOf(v) { return m.t + ph - (v - win.y0) / (ymax - win.y0) * ph; }

    ctx.font = MONO_FONT_SM;
    ctx.lineWidth = 1;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var xt = ticks(xmin, xmax, 5);
    for (i = 0; i < xt.length; i++) {
      var gx = Math.round(xOf(xt[i])) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(gx, m.t);
      ctx.lineTo(gx, m.t + ph);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(fmtInt(xt[i]), gx, m.t + ph + 5);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    var yt = ticks(0, ymax, 4);
    for (i = 0; i < yt.length; i++) {
      var gy = Math.round(yOf(yt[i])) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(m.l, gy);
      ctx.lineTo(m.l + pw, gy);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(yt[i].toFixed(2), m.l - 6, gy);
    }

    /* correlation_peak 阈值虚线 */
    var peak = (sync && isNum(sync.correlation_peak)) ? sync.correlation_peak : null;
    if (peak !== null && peak > 0 && peak <= ymax) {
      ctx.save();
      ctx.strokeStyle = COL.floor;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      var py0 = Math.round(yOf(peak)) + 0.5;
      ctx.moveTo(m.l, py0);
      ctx.lineTo(m.l + pw, py0);
      ctx.stroke();
      ctx.restore();
    }

    /* 相关幅度折线（裁剪到绘图区） */
    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, pw, ph);
    ctx.clip();
    ctx.strokeStyle = COL.line;
    ctx.lineWidth = 1.3;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    var started = false;
    for (i = 0; i < pts.length; i++) {
      var lx = xOf(pts[i].x);
      var ly = yOf(pts[i].y);
      if (!started) { ctx.moveTo(lx, ly); started = true; } else { ctx.lineTo(lx, ly); }
    }
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.restore();

    /* 每帧起点：竖线 + 相关曲线上的圆点 */
    var frames = (d && Array.isArray(d.frames)) ? d.frames : [];
    var marked = 0;
    var step = (xmax - xmin) / Math.max(1, pts.length);
    for (i = 0; i < frames.length; i++) {
      var fs = (frames[i] && isNum(frames[i].start_symbol)) ? frames[i].start_symbol : null;
      if (fs === null || fs < xmin || fs > xmax) { continue; }
      var vx = Math.round(xOf(fs)) + 0.5;
      ctx.save();
      ctx.strokeStyle = COL.resid;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(vx, m.t);
      ctx.lineTo(vx, m.t + ph);
      ctx.stroke();
      ctx.restore();
      var near = -1;
      var bd = Infinity;
      for (var k = 0; k < pts.length; k++) {
        var dd = Math.abs(pts[k].x - fs);
        if (dd < bd) { bd = dd; near = k; }
      }
      if (near >= 0 && bd <= Math.max(1, step * 2)) {
        ctx.beginPath();
        ctx.arc(xOf(pts[near].x), yOf(pts[near].y), 3, 0, Math.PI * 2);
        ctx.fillStyle = COL.resid;
        ctx.fill();
        marked++;
      }
    }
    /* sync.frame_start_symbol：检测到的首帧起点，用真值色加粗标一次 */
    var first = (sync && isNum(sync.frame_start_symbol)) ? sync.frame_start_symbol : null;
    if (first !== null && first >= xmin && first <= xmax) {
      var fx = Math.round(xOf(first)) + 0.5;
      ctx.strokeStyle = COL.true;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(fx, m.t);
      ctx.lineTo(fx, m.t + ph);
      ctx.stroke();
      ctx.lineWidth = 1;
    }

    ctx.fillStyle = COL.axisText2;
    ctx.font = MONO_FONT_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('归一化相关幅度', m.l + 2, m.t + 2);
    ctx.textAlign = 'right';
    ctx.fillText('符号序号', m.l + pw, m.t + ph + 5);
    strokeFrame(ctx, m.l, m.t, pw, ph);
    drawViewBadge(ctx, 'frameCorr', m, pw, ph);

    if (el.frameCorrMeta) {
      var parts = ['有效 ' + fmtInt(pts.length) + ' 点', '符号 ' + fmtInt(xmin) + '…' + fmtInt(xmax)];
      if (peak !== null) { parts.push('correlation_peak ' + peak.toFixed(3)); }
      if (frames.length) { parts.push('帧起点 ' + fmtInt(marked) + '/' + fmtInt(frames.length)); }
      el.frameCorrMeta.textContent = parts.join('  ·  ');
    }
  }


  /* ================= URL 同步 ================= */
  function readUrl() {
    var q = new URLSearchParams(location.search);
    var m = q.get('mock');
    state.mock = (m === '1' || m === 'true');
    if (q.get('dir')) { state.dir = q.get('dir'); }
    if (q.get('path')) { state.path = q.get('path'); }
    var t = q.get('tab');
    if (t && TABS.indexOf(t) >= 0) { state.activeTab = t; }
    /* 只接受下拉框里存在的调制名，保证 select 显示与请求参数一致 */
    var mod = q.get('mod');
    if (mod && DEMOD_MODS.indexOf(mod) >= 0) { state.ui.demodMod = mod; }
  }

  function syncUrl(push) {
    var q = new URLSearchParams();
    if (state.mock) { q.set('mock', '1'); }
    if (state.path) { q.set('path', state.path); }
    if (state.dir && state.dir !== 'data/demo') { q.set('dir', state.dir); }
    if (state.ui.demodMod) { q.set('mod', state.ui.demodMod); }
    if (state.activeTab !== 'spectrum') { q.set('tab', state.activeTab); }
    var qs = q.toString();
    var url = location.pathname + (qs ? ('?' + qs) : '');
    try {
      if (push) { history.pushState({ path: state.path }, '', url); }
      else { history.replaceState({ path: state.path }, '', url); }
    } catch (err) { /* file:// 下忽略 */ }
  }

  /* ================= 渲染：案例 / 概览 / 告警 / 状态 ================= */
  function caseMatches(c) {
    if (!state.filter) { return true; }
    var f = state.filter.toLowerCase();
    var hay = [c.name, c.prefix, c.modulation].join(' ').toLowerCase();
    return hay.indexOf(f) >= 0;
  }

  function renderCases() {
    var host = el.caseList;
    if (!host) { return; }
    clear(host);
    var list = [];
    for (var i = 0; i < state.cases.length; i++) {
      if (caseMatches(state.cases[i])) { list.push(state.cases[i]); }
    }
    if (el.caseCount) {
      el.caseCount.textContent = state.pending.cases ? '加载中'
        : (String(list.length) + ' / ' + String(state.cases.length));
    }
    var emptyMsg = '尚未加载案例列表';
    if (state.pending.cases) { emptyMsg = '正在加载案例列表…'; }
    else if (state.failed.cases) { emptyMsg = '案例列表加载失败'; }
    else if (state.cases.length && !list.length) { emptyMsg = '没有匹配「' + state.filter + '」的案例'; }
    else if (!state.cases.length) { emptyMsg = '目录内没有可用案例'; }
    else { emptyMsg = '尚未加载案例列表'; }
    if (el.caseEmpty) { el.caseEmpty.textContent = emptyMsg; }
    show(el.caseEmpty, list.length === 0);

    for (i = 0; i < list.length; i++) {
      host.appendChild(caseItem(list[i]));
    }
  }

  function caseItem(c) {
    var li = mk('li');
    var btn = mk('button', 'case-item');
    btn.type = 'button';
    var p = caseDataPath(c);
    var active = state.path && p && state.path === p;
    if (active) { btn.className += ' is-active'; }
    btn.title = c.prefix ? String(c.prefix) : String(c.name);
    btn.appendChild(mk('span', 'case-name', c.name ? String(c.name) : '(未命名)'));

    var tags = mk('div', 'case-tags');
    if (c.modulation) { tags.appendChild(mk('span', 'tag tag-mod', String(c.modulation))); }
    if (c.has_truth) { tags.appendChild(mk('span', 'tag tag-truth', '有真值')); }
    else { tags.appendChild(mk('span', 'tag tag-notruth', '无真值')); }
    btn.appendChild(tags);

    var meta = mk('div', 'case-meta');
    if (isNum(c.snr_db)) { meta.appendChild(mk('span', null, 'SNR ' + c.snr_db.toFixed(1) + ' dB')); }
    if (isNum(c.sample_rate)) { meta.appendChild(mk('span', null, 'Fs ' + fmtCompact(c.sample_rate) + 'Hz')); }
    if (isNum(c.num_samples)) { meta.appendChild(mk('span', null, fmtInt(c.num_samples) + ' pts')); }
    if (isNum(c.duration_s)) { meta.appendChild(mk('span', null, c.duration_s.toFixed(3) + ' s')); }
    btn.appendChild(meta);

    btn.addEventListener('click', function () {
      if (p && state.path !== p) { loadPath(p, { push: true }); }
    });
    li.appendChild(btn);
    return li;
  }

  function renderOverview() {
    var a = state.analyze;
    var s = state.spectrum;
    setText('ovPath', state.path || '—');
    if (!a) {
      var busy = state.pending.analyze || state.pending.spectrum;
      var t = busy ? '加载中…' : (state.failed.analyze ? '加载失败' : '—');
      var ids0 = ['ovFs', 'ovN', 'ovDur', 'ovPeak', 'ovPeakSnr', 'ovBw', 'ovFloor'];
      for (var i = 0; i < ids0.length; i++) { setText(ids0[i], t); setDim(ids0[i], true); }
      return;
    }
    setText('ovFs', isNum(a.sample_rate) ? (fmtCompact(a.sample_rate) + 'Hz') : '—');
    setText('ovN', isNum(a.num_samples) ? fmtInt(a.num_samples) : '—');
    setText('ovDur', isNum(a.duration_s) ? (a.duration_s.toFixed(3) + ' s') : '—');
    var pk = a.peak || {};
    setText('ovPeak', isNum(pk.frequency_hz) ? (fmtCompact(pk.frequency_hz) + 'Hz') : '—');
    setText('ovPeakSnr', isNum(pk.snr_db) ? (pk.snr_db.toFixed(2) + ' dB') : '—');
    var oc = a.occupied || {};
    setText('ovBw', isNum(oc.bandwidth) ? (fmtCompact(oc.bandwidth) + 'Hz') : '—');
    var floor = isNum(a.noise_floor_psd) ? a.noise_floor_psd : (s ? s.noise_floor_db : null);
    setText('ovFloor', isNum(floor) ? (a.noise_floor_psd === floor ? floor.toExponential(2) : floor.toFixed(2) + ' dB') : '—');
    var ids1 = ['ovFs', 'ovN', 'ovDur', 'ovPeak', 'ovPeakSnr', 'ovBw', 'ovFloor'];
    for (i = 0; i < ids1.length; i++) { setDim(ids1[i], false); }
    setDim('ovPath', !state.path);
  }

  function alertLevel(text, fallback) {
    var t = String(text);
    if (t.indexOf('失败') >= 0 || t.indexOf('错误') >= 0 || t.indexOf('异常') >= 0) { return 'err'; }
    if (t.indexOf('低') >= 0 || t.indexOf('偏') >= 0 || t.indexOf('可能') >= 0 || t.indexOf('仅') >= 0) { return 'warn'; }
    return fallback || 'warn';
  }

  function collectAlerts() {
    var out = [];
    var a = state.analyze;
    var i;
    if (a && Array.isArray(a.warnings)) {
      for (i = 0; i < a.warnings.length; i++) { out.push({ level: alertLevel(a.warnings[i], 'warn'), src: 'analyze', text: String(a.warnings[i]) }); }
    }
    var f = state.features;
    if (f && Array.isArray(f.warnings)) {
      for (i = 0; i < f.warnings.length; i++) { out.push({ level: alertLevel(f.warnings[i], 'warn'), src: 'features', text: String(f.warnings[i]) }); }
    }
    var c = state.classify;
    if (c) {
      if (c.gate === 'reject') { out.push({ level: 'err', src: 'classify', text: '门控判定为拒识（reject），分类结果不可信' }); }
      if (c.ood && c.ood.is_outlier === true) { out.push({ level: 'warn', src: 'classify', text: 'OOD 判定为离群样本' + (isNum(c.ood.score) ? ('（score ' + c.ood.score.toFixed(2) + '）') : '') }); }
      if (Array.isArray(c.warnings)) {
        for (i = 0; i < c.warnings.length; i++) { out.push({ level: alertLevel(c.warnings[i], 'warn'), src: 'classify', text: String(c.warnings[i]) }); }
      }
    }
    var dm = state.demod;
    if (dm) {
      var lk = dm.lock || {};
      if (lk.timing === false) { out.push({ level: 'err', src: 'demod', text: '定时环路失锁（lock.timing = false）' }); }
      else if (isNum(lk.timing_metric) && lk.timing_metric < 0.4) { out.push({ level: 'warn', src: 'demod', text: '定时环路临界：timing_metric ' + lk.timing_metric.toFixed(3) }); }
      if (lk.carrier === false) { out.push({ level: 'err', src: 'demod', text: '载波环路失锁（lock.carrier = false）' }); }
      else if (isNum(lk.carrier_metric) && lk.carrier_metric < 0.4) { out.push({ level: 'warn', src: 'demod', text: '载波环路临界：carrier_metric ' + lk.carrier_metric.toFixed(3) }); }
      if (Array.isArray(dm.warnings)) {
        for (i = 0; i < dm.warnings.length; i++) { out.push({ level: alertLevel(dm.warnings[i], 'warn'), src: 'demod', text: String(dm.warnings[i]) }); }
      }
    }
    var fr = state.frame;
    if (fr) {
      var fsync = (fr.sync && typeof fr.sync === 'object') ? fr.sync : null;
      if (fsync && fsync.locked === false) { out.push({ level: 'err', src: 'frame', text: '帧同步未锁定（sync.locked = false）' }); }
      if (Array.isArray(fr.warnings)) {
        for (i = 0; i < fr.warnings.length; i++) { out.push({ level: alertLevel(fr.warnings[i], 'warn'), src: 'frame', text: String(fr.warnings[i]) }); }
      }
      var fsumm = (fr.summary && typeof fr.summary === 'object') ? fr.summary : null;
      if (fsumm && isNum(fsumm.frames) && isNum(fsumm.frames_crc_ok) && fsumm.frames_crc_ok < fsumm.frames) {
        out.push({ level: 'warn', src: 'frame', text: 'CRC 未通过帧 ' + fmtInt(fsumm.frames - fsumm.frames_crc_ok) + ' / ' + fmtInt(fsumm.frames) });
      }
    }
    var sc = state.ssca;
    if (sc && Array.isArray(sc.warnings)) {
      for (i = 0; i < sc.warnings.length; i++) { out.push({ level: alertLevel(sc.warnings[i], 'warn'), src: 'ssca', text: String(sc.warnings[i]) }); }
    }
    var nr = state.narrate;
    if (nr && Array.isArray(nr.warnings)) {
      for (i = 0; i < nr.warnings.length; i++) { out.push({ level: alertLevel(nr.warnings[i], 'warn'), src: 'narrate', text: String(nr.warnings[i]) }); }
    }
    return out;
  }

  function renderAlerts() {
    var list = collectAlerts();
    var host = el.alertList;
    if (!host) { return; }
    clear(host);
    if (!state.path) { show(el.alertStrip, false); return; }
    show(el.alertStrip, true);
    if (!list.length) {
      var ok = mk('li', 'alert-item alert-ok');
      ok.appendChild(mk('span', null, '无告警'));
      host.appendChild(ok);
      return;
    }
    for (var i = 0; i < list.length; i++) {
      var n = mk('li', 'alert-item alert-' + list[i].level);
      n.appendChild(mk('span', 'a-src', list[i].src));
      n.appendChild(mk('span', null, list[i].text));
      host.appendChild(n);
    }
  }

  function anyPending() {
    for (var k in state.pending) {
      if (Object.prototype.hasOwnProperty.call(state.pending, k) && state.pending[k]) { return true; }
    }
    return false;
  }

  function renderStatus() {
    var busy = anyPending();
    setText('statusPath', state.path ? state.path : '未选择数据');
    if (el.statusPath) {
      el.statusPath.title = state.path ? state.path : '';
      el.statusPath.className = 'status-item mono' + (busy ? ' is-busy' : (state.errors.length ? ' is-err' : ''));
    }
    setText('statusTab', '视图 ' + state.activeTab);
    setText('statusUpdated', busy ? '加载中…' : (state.updatedAt ? ('更新 ' + state.updatedAt) : '—'));
    setText('statusSource', state.mock ? 'source: mock' : 'source: backend');
  }

  function renderAll() {
    renderOverview();
    renderAlerts();
    renderSpectrumLegend();
    renderEstimate();
    renderFeatures();
    renderClassify();
    renderDemod();
    renderFec();
    renderFrame();
    renderSsca();
    renderNarrate();
    renderAgent();
    renderLlmSettings();
    renderAuthStatus();
    redrawVisible();
    renderStatus();
  }

  /* ================= Canvas 基础设施 ================= */
  function fitCanvas(canvas) {
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(1, Math.round(rect.width));
    var h = Math.max(1, Math.round(rect.height));
    var pw = Math.round(w * dpr);
    var ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h, dpr: dpr };
  }

  function strokeFrame(ctx, x, y, w, h) {
    ctx.strokeStyle = COL.frame;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  }

  function getOffscreen(w, h) {
    if (!offCanvas) { offCanvas = document.createElement('canvas'); }
    if (offCanvas.width !== w || offCanvas.height !== h) { offCanvas.width = w; offCanvas.height = h; }
    return offCanvas;
  }

  function buildColormap(stops) {
    var lut = new Array(256);
    for (var i = 0; i < 256; i++) {
      var t = i / 255 * (stops.length - 1);
      var a = Math.floor(t);
      var b = Math.min(a + 1, stops.length - 1);
      var f = t - a;
      lut[i] = [
        Math.round(stops[a][0] + (stops[b][0] - stops[a][0]) * f),
        Math.round(stops[a][1] + (stops[b][1] - stops[a][1]) * f),
        Math.round(stops[a][2] + (stops[b][2] - stops[a][2]) * f)
      ];
    }
    return lut;
  }

  function cmap(t) {
    if (!COLORMAP) {
      COLORMAP = buildColormap([[7, 11, 16], [14, 42, 61], [20, 101, 122], [34, 179, 154], [255, 207, 107]]);
    }
    var i = Math.round(t * 255);
    if (i < 0) { i = 0; }
    if (i > 255) { i = 255; }
    return COLORMAP[i];
  }

  function rgbCss(c) { return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')'; }

  /* ================= 交互式图表层（统一缩放 / 拖动 / 复位） =================
   * 全部 canvas 图共用一套 view 状态与事件，图自己只声明"完整数据域 + 轴性质 + 重绘函数"。
   * 关键约定：
   *   - 窗口一律存"显示坐标"：线性轴 = 数据值，对数轴 = log10(数据值)。
   *     于是缩放/平移/钳制只有一套算术，对数轴天然按对数缩放（不会被线性拉伸）。
   *   - 每次绘制调用 chartWindow()，它用 full 域的签名判断数据是否刷新；
   *     签名变化 → 窗口复位（满足"缩放状态在数据刷新时重置"）。
   *   - 最小跨度 = max(8 个数据点, 1e-3 × 原始范围)，最大不超过原始范围。
   */
  var VIEWS = {};
  var dragV = null;

  function viewOf(id) {
    if (!VIEWS[id]) {
      VIEWS[id] = {
        id: id, canvas: null, redraw: null, bound: false, btn: null,
        m: null, pw: 0, ph: 0,
        full: null, win: null, sig: '', zoomed: false,
        xLog: false, yLog: false, xCount: 0, yCount: 0, yZoom: true,
        fmtX: null, fmtY: null, unitX: '', unitY: '', drag: null
      };
    }
    return VIEWS[id];
  }

  function l10(v) { return Math.log10(v > 1e-12 ? v : 1e-12); }
  function pow10(v) { return Math.pow(10, v); }
  function toDisp(v, log) { return log ? l10(v) : v; }
  function toRaw(v, log) { return log ? pow10(v) : v; }

  function minSpanD(fullSpan, count) {
    var byRatio = fullSpan * 1e-3;
    var byPts = (count > 1) ? (fullSpan * 8 / (count - 1)) : byRatio;
    var m = byRatio > byPts ? byRatio : byPts;
    /* 数据点太少时不能因为"8 个点"反而把图钉死，退回 1e-3 档，最多只占一半视野 */
    if (m > fullSpan * 0.5) { m = Math.max(byRatio, fullSpan * 0.5); }
    if (m > fullSpan) { m = fullSpan; }
    return m;
  }

  function clampAxis(w0, w1, f0, f1, count) {
    var full = f1 - f0;
    var span = w1 - w0;
    if (!(span > 0) || !(full > 0) || span >= full) { return [f0, f1]; }
    var minS = minSpanD(full, count);
    if (span < minS) { span = minS; }
    if (w0 < f0) { w0 = f0; }
    if (w0 + span > f1) { w0 = f1 - span; }
    if (w0 < f0) { w0 = f0; }
    return [w0, w0 + span];
  }

  function clampView(v) {
    var f = v.full;
    var w = v.win;
    var x = clampAxis(w.x0, w.x1, f.x0, f.x1, v.xCount);
    w.x0 = x[0]; w.x1 = x[1];
    var y = clampAxis(w.y0, w.y1, f.y0, f.y1, v.yCount);
    w.y0 = y[0]; w.y1 = y[1];
  }

  function isZoomedV(v) {
    var f = v.full;
    var w = v.win;
    var ex = (f.x1 - f.x0) * 1e-9;
    var ey = (f.y1 - f.y0) * 1e-9;
    if (Math.abs(w.x0 - f.x0) > ex || Math.abs(w.x1 - f.x1) > ex) { return true; }
    if (v.yZoom && (Math.abs(w.y0 - f.y0) > ey || Math.abs(w.y1 - f.y1) > ey)) { return true; }
    return false;
  }

  /* 图为数据刷新后必须复位窗口；用 full 域签名判断，避免每个 loader 都要记着一句 reset */
  function chartWindow(id, full, opts) {
    opts = opts || {};
    var v = viewOf(id);
    v.xLog = !!opts.xLog;
    v.yLog = !!opts.yLog;
    v.xCount = opts.xCount || 0;
    v.yCount = opts.yCount || 0;
    v.yZoom = opts.yZoom !== false;
    if (opts.fmtX) { v.fmtX = opts.fmtX; }
    if (opts.fmtY) { v.fmtY = opts.fmtY; }
    if (opts.unitX !== undefined) { v.unitX = opts.unitX; }
    if (opts.unitY !== undefined) { v.unitY = opts.unitY; }
    if (opts.redraw) { v.redraw = opts.redraw; }
    if (opts.canvas) { v.canvas = opts.canvas; }

    var x0 = isNum(full.x0) ? full.x0 : 0;
    var x1 = isNum(full.x1) ? full.x1 : 1;
    var y0 = isNum(full.y0) ? full.y0 : 0;
    var y1 = isNum(full.y1) ? full.y1 : 1;
    if (!(x1 > x0)) { x1 = x0 + 1; }
    if (!(y1 > y0)) { y1 = y0 + 1; }

    var fd = { x0: toDisp(x0, v.xLog), x1: toDisp(x1, v.xLog), y0: toDisp(y0, v.yLog), y1: toDisp(y1, v.yLog) };
    var sig = fd.x0.toPrecision(8) + '|' + fd.x1.toPrecision(8) + '|' + fd.y0.toPrecision(8) + '|' + fd.y1.toPrecision(8)
      + '|' + (v.xLog ? 'L' : 'l') + (v.yLog ? 'L' : 'l') + '|' + v.xCount + '|' + v.yCount;
    if (sig !== v.sig) { v.sig = sig; v.win = null; }
    v.full = fd;
    if (!v.win) { v.win = { x0: fd.x0, x1: fd.x1, y0: fd.y0, y1: fd.y1 }; }
    else { clampView(v); }
    v.zoomed = isZoomedV(v);
    if (v.canvas && !v.bound) { chartBind(v); }
    updateResetBtn(v);
    return {
      x0: toRaw(v.win.x0, v.xLog), x1: toRaw(v.win.x1, v.xLog),
      y0: toRaw(v.win.y0, v.yLog), y1: toRaw(v.win.y1, v.yLog)
    };
  }

  /* 绘图区几何（CSS 像素），滚轮/拖动换算锚点用 */
  function chartGeom(id, m, pw, ph) {
    var v = viewOf(id);
    v.m = m; v.pw = pw; v.ph = ph;
  }

  function fmtRangeNum(x) {
    if (!isNum(x)) { return '—'; }
    var a = Math.abs(x);
    if (a === 0) { return '0'; }
    if (a >= 1e4 || a < 1e-3) { return x.toExponential(2); }
    if (a >= 100) { return x.toFixed(0); }
    if (a >= 1) { return x.toFixed(2); }
    return x.toFixed(4);
  }

  function viewRangeText(id) {
    var v = viewOf(id);
    if (!v.win || !v.full) { return ''; }
    var fx = v.fmtX || fmtRangeNum;
    var fy = v.fmtY || fmtRangeNum;
    var x0 = toRaw(v.win.x0, v.xLog);
    var x1 = toRaw(v.win.x1, v.xLog);
    var y0 = toRaw(v.win.y0, v.yLog);
    var y1 = toRaw(v.win.y1, v.yLog);
    return 'X ' + fx(x0) + '…' + fx(x1) + (v.unitX ? (' ' + v.unitX) : '')
      + ', Y ' + fy(y0) + '…' + fy(y1) + (v.unitY ? (' ' + v.unitY) : '');
  }

  /* 缩放后在图内右下角显示当前范围（轴刻度本来就已经跟随窗口） */
  function drawViewBadge(ctx, id, m, pw, ph) {
    var v = viewOf(id);
    if (!v.zoomed || !v.win) { return; }
    var txt = viewRangeText(id);
    if (!txt) { return; }
    ctx.save();
    ctx.font = MONO_FONT_SM;
    var tw = ctx.measureText(txt).width;
    var bx = m.l + pw - tw - 10;
    var by = m.t + ph - 17;
    if (bx < m.l + 2) { bx = m.l + 2; }
    if (by < m.t + 2) { by = m.t + 2; }
    ctx.fillStyle = 'rgba(10, 14, 19, 0.86)';
    ctx.fillRect(bx - 4, by - 3, tw + 8, 15);
    ctx.strokeStyle = COL.grid;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx - 3.5, by - 2.5, tw + 7, 14);
    ctx.fillStyle = COL.axisText2;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(txt, bx, by);
    ctx.restore();
  }

  function chartDragging() { return !!(dragV && dragV.drag && dragV.drag.moved); }

  function chartBind(v) {
    v.bound = true;
    var cv = v.canvas;
    if (!cv) { return; }
    cv.__chartId = v.id;
    cv.addEventListener('wheel', chartOnWheel, { passive: false });
    cv.addEventListener('mousedown', chartOnDown);
    cv.addEventListener('dblclick', chartOnDblClick);
    ensureResetBtn(v);
  }

  function ensureResetBtn(v) {
    if (v.btn || !v.canvas) { return; }
    var host = v.canvas.parentNode;
    if (!host || !host.appendChild) { return; }
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'chart-reset';
    b.textContent = '重置视图';
    b.setAttribute('aria-label', '重置视图');
    b.addEventListener('click', function (ev) { ev.preventDefault(); chartReset(v.id); });
    host.appendChild(b);
    v.btn = b;
  }

  function updateResetBtn(v) {
    if (!v.btn) { return; }
    v.btn.className = 'chart-reset' + (v.zoomed ? ' is-active' : '');
    v.btn.title = (v.zoomed ? ('当前视图 ' + viewRangeText(v.id) + '\n') : '')
      + '重置视图（双击图表同样复位）；滚轮以鼠标为锚点缩放，Shift 只缩 X，Alt 只缩 Y，按住左键拖动平移';
  }

  function chartReset(id) {
    var v = viewOf(id);
    if (!v.full) { return; }
    v.win = { x0: v.full.x0, x1: v.full.x1, y0: v.full.y0, y1: v.full.y1 };
    v.zoomed = false;
    updateResetBtn(v);
    if (v.redraw) { v.redraw(); }
  }

  function zoomAxis(w, ax, frac, k) {
    var a = ax === 'x' ? 'x0' : 'y0';
    var b = ax === 'x' ? 'x1' : 'y1';
    var span = w[b] - w[a];
    var nspan = span / k;
    if (!isFinite(nspan) || !(nspan > 0)) { return; }
    var anchor = w[a] + span * frac;
    w[a] = anchor - nspan * frac;
    w[b] = w[a] + nspan;
  }

  function chartZoomTo(id, ev, k, only) {
    var v = viewOf(id);
    if (!v.canvas || !v.m || !v.win) { return; }
    if (only === 'y' && !v.yZoom) { return; }
    if (only === 'both' && !v.yZoom) { only = 'x'; }
    var rect = v.canvas.getBoundingClientRect();
    var fx = (ev.clientX - rect.left - v.m.l) / v.pw;
    var fy = (v.m.t + v.ph - (ev.clientY - rect.top)) / v.ph;
    if (!isFinite(fx)) { fx = 0.5; }
    if (!isFinite(fy)) { fy = 0.5; }
    fx = Math.min(1, Math.max(0, fx));
    fy = Math.min(1, Math.max(0, fy));
    if (only !== 'y') { zoomAxis(v.win, 'x', fx, k); }
    if (only !== 'x') { zoomAxis(v.win, 'y', fy, k); }
    clampView(v);
    v.zoomed = isZoomedV(v);
    updateResetBtn(v);
    if (v.redraw) { v.redraw(); }
  }

  function chartOnWheel(ev) {
    var v = viewOf(this.__chartId);
    if (!v || !v.canvas || !v.m || !v.win) { return; }
    ev.preventDefault();
    var step = (ev.deltaMode === 1) ? 16 : 1;
    var k = Math.pow(1.0015, -ev.deltaY * step);
    if (!(k > 0) || k === 1) { return; }
    var only = ev.shiftKey ? 'x' : (ev.altKey ? 'y' : 'both');
    chartZoomTo(this.__chartId, ev, k, only);
  }

  function chartOnDown(ev) {
    if (ev.button !== 0) { return; }
    var v = viewOf(this.__chartId);
    if (!v || !v.canvas || !v.m || !v.win) { return; }
    dragV = v;
    v.drag = {
      x: ev.clientX, y: ev.clientY, moved: false,
      w: { x0: v.win.x0, x1: v.win.x1, y0: v.win.y0, y1: v.win.y1 }
    };
    try { v.canvas.style.cursor = 'grabbing'; } catch (e) { /* 忽略 */ }
    document.addEventListener('mousemove', chartOnDrag);
    document.addEventListener('mouseup', chartOnUp);
    ev.preventDefault();
  }

  function chartOnDrag(ev) {
    var v = dragV;
    if (!v || !v.drag || !v.m || !v.win) { return; }
    var d = v.drag;
    var dx = ev.clientX - d.x;
    var dy = ev.clientY - d.y;
    if (!d.moved && (Math.abs(dx) + Math.abs(dy)) < 3) { return; }
    d.moved = true;
    var spanX = d.w.x1 - d.w.x0;
    var spanY = d.w.y1 - d.w.y0;
    v.win.x0 = d.w.x0 - dx / v.pw * spanX;
    v.win.x1 = v.win.x0 + spanX;
    if (v.yZoom) {
      v.win.y0 = d.w.y0 + dy / v.ph * spanY;
      v.win.y1 = v.win.y0 + spanY;
    }
    clampView(v);
    v.zoomed = isZoomedV(v);
    updateResetBtn(v);
    if (v.redraw) { v.redraw(); }
    ev.preventDefault();
  }

  function chartOnUp() {
    if (dragV) {
      try { if (dragV.canvas && dragV.canvas.style) { dragV.canvas.style.cursor = ''; } } catch (e) { /* 忽略 */ }
      dragV.drag = null;
    }
    dragV = null;
    document.removeEventListener('mousemove', chartOnDrag);
    document.removeEventListener('mouseup', chartOnUp);
  }

  function chartOnDblClick(ev) {
    ev.preventDefault();
    chartReset(this.__chartId);
  }

  /* ================= 频谱图 ================= */
  function drawSpectrum() {
    var cv = el.spectrumCanvas;
    if (!cv) { return; }
    var emp = el.spectrumEmpty;
    var d = state.spectrum;
    var ok = d && Array.isArray(d.freqs) && Array.isArray(d.psd_db) && d.freqs.length > 1 && d.psd_db.length > 1;
    if (!ok) {
      if (emp) {
        emp.textContent = state.pending.spectrum ? '加载中…'
          : (state.failed.spectrum ? '频谱加载失败（见顶部错误条）'
            : (state.path ? '暂无频谱数据' : '等待数据：请在左侧选择案例或输入路径'));
        show(emp, true);
      }
      return;
    }
    if (emp) { show(emp, false); }

    var f = fitCanvas(cv);
    var ctx = f.ctx;
    var scale = state.ui.scale;
    var residual = state.ui.residual && Array.isArray(d.residual_db) && d.residual_db.length === d.psd_db.length;
    var m = { l: 58, r: residual ? 46 : 14, t: 12, b: 22 };
    var pw = f.w - m.l - m.r;
    var ph = f.h - m.t - m.b;
    if (pw < 24 || ph < 24) { return; }

    var fx0 = d.freqs[0];
    var fx1 = d.freqs[d.freqs.length - 1];
    if (!isNum(fx0) || !isNum(fx1) || fx1 <= fx0) { fx0 = 0; fx1 = 1; }

    var i;
    var series;
    var ymin;
    var ymax;
    if (scale === 'linear') {
      series = new Array(d.psd_db.length);
      var mx = 0;
      for (i = 0; i < d.psd_db.length; i++) {
        var v = isNum(d.psd_db[i]) ? Math.pow(10, d.psd_db[i] / 10) : 0;
        series[i] = v;
        if (v > mx) { mx = v; }
      }
      ymin = 0;
      ymax = mx > 0 ? mx * 1.10 : 1;
    } else {
      series = d.psd_db;
      var lo = percentile(series, 0.005);
      var hi = percentile(series, 0.995);
      var realMax = -Infinity;
      var realMin = Infinity;
      for (i = 0; i < series.length; i++) {
        if (!isNum(series[i])) { continue; }
        if (series[i] > realMax) { realMax = series[i]; }
        if (series[i] < realMin) { realMin = series[i]; }
      }
      if (!isNum(lo)) { lo = realMin; }
      if (!isNum(hi)) { hi = realMax; }
      if (realMax > hi) { hi = realMax; }
      if (hi - lo < 1e-9) { hi = lo + 1; }
      var pad = (hi - lo) * 0.08;
      ymin = lo - pad;
      ymax = hi + pad;
    }

    /* 交互层：X = 频率，Y = PSD。dB 轴本身就是对数刻度，窗口按 dB 线性收缩即功率对数缩放 */
    var win = chartWindow('spectrum', { x0: fx0, x1: fx1, y0: ymin, y1: ymax }, {
      xCount: d.freqs.length, yCount: 0,
      fmtX: fmtHzTick, unitX: 'Hz',
      fmtY: function (v) { return v.toFixed(1); },
      unitY: (scale === 'linear') ? '' : 'dB',
      canvas: cv, redraw: drawSpectrum
    });
    fx0 = win.x0; fx1 = win.x1; ymin = win.y0; ymax = win.y1;
    chartGeom('spectrum', m, pw, ph);

    function xOf(freq) { return m.l + (freq - fx0) / (fx1 - fx0) * pw; }
    function yOf(val) { return m.t + ph - (val - ymin) / (ymax - ymin) * ph; }

    /* 占用带宽阴影 */
    var oc = d.occupied || {};
    if (isNum(oc.f_low) && isNum(oc.f_high)) {
      var bx0 = Math.max(m.l, xOf(oc.f_low));
      var bx1 = Math.min(m.l + pw, xOf(oc.f_high));
      if (bx1 > bx0) {
        ctx.fillStyle = COL.band;
        ctx.fillRect(bx0, m.t, bx1 - bx0, ph);
        ctx.strokeStyle = COL.bandEdge;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(Math.round(bx0) + 0.5, m.t);
        ctx.lineTo(Math.round(bx0) + 0.5, m.t + ph);
        ctx.moveTo(Math.round(bx1) + 0.5, m.t);
        ctx.lineTo(Math.round(bx1) + 0.5, m.t + ph);
        ctx.stroke();
      }
    }

    /* 网格与刻度 */
    ctx.font = MONO_FONT_SM;
    ctx.lineWidth = 1;
    var xt = ticks(fx0, fx1, 6);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (i = 0; i < xt.length; i++) {
      var xx = Math.round(xOf(xt[i])) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(xx, m.t);
      ctx.lineTo(xx, m.t + ph);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(fmtHzTick(xt[i]), xx, m.t + ph + 5);
    }
    var yt = ticks(ymin, ymax, 4);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (i = 0; i < yt.length; i++) {
      var yy = Math.round(yOf(yt[i])) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(m.l, yy);
      ctx.lineTo(m.l + pw, yy);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(scale === 'linear' ? yt[i].toExponential(1) : yt[i].toFixed(1), m.l - 6, yy);
    }

    /* 噪声底虚线 */
    var floorDb = isNum(d.noise_floor_db) ? d.noise_floor_db : null;
    if (floorDb !== null) {
      var fv = scale === 'linear' ? Math.pow(10, floorDb / 10) : floorDb;
      if (fv >= ymin && fv <= ymax) {
        ctx.save();
        ctx.strokeStyle = COL.floor;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        var fy = Math.round(yOf(fv)) + 0.5;
        ctx.moveTo(m.l, fy);
        ctx.lineTo(m.l + pw, fy);
        ctx.stroke();
        ctx.restore();
        ctx.fillStyle = COL.floor;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText('噪声底 ' + floorDb.toFixed(2) + ' dB', m.l + 6, fy - 2);
      }
    }

    /* 残余曲线（右轴 -60..0 dB） */
    var resid = residual && Array.isArray(d.residual_db);
    if (resid) {
      ctx.strokeStyle = COL.resid;
      ctx.lineWidth = 1;
      ctx.save();
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      var started = false;
      for (i = 0; i < d.residual_db.length; i++) {
        var rv = d.residual_db[i];
        if (!isNum(rv)) { started = false; continue; }
        var t2 = (rv + 60) / 60;
        if (t2 < 0) { t2 = 0; }
        if (t2 > 1) { t2 = 1; }
        var px = xOf(d.freqs[i]);
        var py = m.t + ph - t2 * ph;
        if (!started) { ctx.moveTo(px, py); started = true; } else { ctx.lineTo(px, py); }
      }
      ctx.stroke();
      ctx.restore();
      var rx = m.l + pw + 6;
      ctx.fillStyle = COL.resid;
      ctx.font = MONO_FONT_SM;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('0', rx, m.t + 4);
      ctx.fillText('-60', rx, m.t + ph - 4);
      ctx.textBaseline = 'top';
      ctx.fillText('dB', rx, m.t + 12);
    }

    /* PSD 折线 + 面积（裁剪到绘图区，缩放/平移后不越界） */
    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, pw, ph);
    ctx.clip();
    ctx.beginPath();
    var first = true;
    for (i = 0; i < series.length; i++) {
      if (!isNum(series[i])) { continue; }
      var lx = xOf(d.freqs[i]);
      var ly = yOf(series[i]);
      if (first) { ctx.moveTo(lx, ly); first = false; } else { ctx.lineTo(lx, ly); }
    }
    ctx.strokeStyle = COL.line;
    ctx.lineWidth = 1.4;
    ctx.lineJoin = 'round';
    ctx.stroke();
    ctx.save();
    ctx.lineTo(m.l + pw, m.t + ph);
    ctx.lineTo(m.l, m.t + ph);
    ctx.closePath();
    ctx.fillStyle = COL.fill;
    ctx.fill();
    ctx.restore();
    ctx.restore();

    /* 峰值标记 */
    var pk = d.peak || {};
    if (isNum(pk.frequency_hz)) {
      var pi = nearestIndex(d.freqs, pk.frequency_hz);
      if (pi >= 0 && isNum(series[pi])) {
        var pxp = xOf(d.freqs[pi]);
        var pyp = yOf(series[pi]);
        ctx.beginPath();
        ctx.arc(pxp, pyp, 2.6, 0, Math.PI * 2);
        ctx.fillStyle = COL.peak;
        ctx.fill();
      }
    }

    /* 载频估计 / 真值竖线 */
    function vline(freq, color, label, dash) {
      if (!isNum(freq)) { return; }
      var x = xOf(freq);
      if (x < m.l - 1 || x > m.l + pw + 1) { return; }
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      if (dash) { ctx.setLineDash([4, 3]); }
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, m.t);
      ctx.lineTo(Math.round(x) + 0.5, m.t + ph);
      ctx.stroke();
      ctx.restore();
      ctx.font = MONO_FONT_SM;
      ctx.textBaseline = 'top';
      var tw = ctx.measureText(label).width;
      var tx = x + 4;
      if (tx + tw > m.l + pw) { tx = x - 4 - tw; }
      if (tx < m.l) { tx = m.l + 2; }
      ctx.fillStyle = color;
      ctx.textAlign = 'left';
      ctx.fillText(label, tx, m.t + 3);
    }
    var cfoEst = (d.cfo && isNum(d.cfo.value)) ? d.cfo.value : null;
    var cfoTrue = isNum(d.cfo_true) ? d.cfo_true : null;
    vline(cfoEst, COL.line, '估计 ' + fmtHzTick(cfoEst) + 'Hz', true);
    vline(cfoTrue, COL.true, '真值 ' + fmtHzTick(cfoTrue) + 'Hz', false);

    /* 轴标注 */
    ctx.fillStyle = COL.axisText2;
    ctx.font = MONO_FONT_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(scale === 'linear' ? '功率 (线性)' : 'PSD (dB)', m.l + 2, m.t + 2);
    ctx.textAlign = 'right';
    ctx.fillText('频率 (Hz)', m.l + pw, m.t + ph + 5);

    strokeFrame(ctx, m.l, m.t, pw, ph);
    drawViewBadge(ctx, 'spectrum', m, pw, ph);
  }

  function nearestIndex(arr, target) {
    var best = -1;
    var bd = Infinity;
    for (var i = 0; i < arr.length; i++) {
      if (!isNum(arr[i])) { continue; }
      var d = Math.abs(arr[i] - target);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  function renderSpectrumLegend() {
    var host = el.spectrumLegend;
    if (!host) { return; }
    clear(host);
    var d = state.spectrum;
    var items = [['legend-swatch', 'PSD']];
    if (d && isNum(d.noise_floor_db)) { items.push(['legend-swatch is-dash', '噪声底']); }
    if (d && d.occupied && isNum(d.occupied.f_low)) { items.push(['legend-swatch is-band', '占用带宽']); }
    if (d && d.cfo && isNum(d.cfo.value)) { items.push(['legend-swatch', '估计 CFO']); }
    if (d && isNum(d.cfo_true)) { items.push(['legend-swatch is-true', '真值 CFO']); }
    if (state.ui.residual) { items.push(['legend-swatch is-resid', '残余 (右轴)']); }
    for (var i = 0; i < items.length; i++) {
      var n = mk('span', 'legend-item');
      n.appendChild(mk('span', items[i][0]));
      n.appendChild(mk('span', null, items[i][1]));
      host.appendChild(n);
    }
  }

  /* ================= 瀑布图 ================= */
  function drawWaterfall() {
    var cv = el.waterfallCanvas;
    if (!cv) { return; }
    var emp = el.waterfallEmpty;
    var d = state.waterfall;
    var ok = d && Array.isArray(d.magnitude_db) && d.magnitude_db.length > 0
      && Array.isArray(d.freqs) && Array.isArray(d.times) && Array.isArray(d.magnitude_db[0])
      && d.magnitude_db[0].length > 0;
    if (!ok) {
      if (emp) {
        emp.textContent = state.pending.waterfall ? '加载中…'
          : (state.failed.waterfall ? '瀑布加载失败（见顶部错误条）'
            : (state.path ? '暂无瀑布数据' : '等待数据：请在左侧选择案例或输入路径'));
        show(emp, true);
      }
      if (el.waterfallMeta) { el.waterfallMeta.textContent = '—'; }
      return;
    }
    if (emp) { show(emp, false); }

    var F = d.magnitude_db.length;
    var T = d.magnitude_db[0].length;
    var minDb = isNum(d.min_db) ? d.min_db : -80;
    var maxDb = isNum(d.max_db) ? d.max_db : -20;
    if (maxDb <= minDb) { maxDb = minDb + 1; }

    if (el.waterfallMeta) {
      el.waterfallMeta.textContent = 'shape ' + F + '×' + T + '  ·  ' + minDb.toFixed(1)
        + ' … ' + maxDb.toFixed(1) + ' dB  ·  ' + (d.times.length ? d.times[d.times.length - 1].toFixed(3) : '0') + ' s';
    }

    var off = getOffscreen(T, F);
    var octx = off.getContext('2d');
    var img = octx.createImageData(T, F);
    var r;
    var c;
    for (r = 0; r < F; r++) {
      var row = d.magnitude_db[r];
      var iy = F - 1 - r;
      for (c = 0; c < T; c++) {
        var v = row ? row[c] : null;
        var t = isNum(v) ? (v - minDb) / (maxDb - minDb) : 0;
        if (t < 0) { t = 0; }
        if (t > 1) { t = 1; }
        var rgb = cmap(t);
        var p = (iy * T + c) * 4;
        img.data[p] = rgb[0];
        img.data[p + 1] = rgb[1];
        img.data[p + 2] = rgb[2];
        img.data[p + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);

    var f = fitCanvas(cv);
    var ctx = f.ctx;
    var m = { l: 58, r: 62, t: 12, b: 22 };
    var pw = f.w - m.l - m.r;
    var ph = f.h - m.t - m.b;
    if (pw < 24 || ph < 24) { return; }

    var fxAll0 = d.freqs[0];
    var fxAll1 = d.freqs[d.freqs.length - 1];
    if (!isNum(fxAll0) || !isNum(fxAll1) || fxAll1 <= fxAll0) { fxAll0 = 0; fxAll1 = 1; }
    var txAll0 = isNum(d.times[0]) ? d.times[0] : 0;
    var txAll1 = isNum(d.times[d.times.length - 1]) ? d.times[d.times.length - 1] : 0;
    if (txAll1 <= txAll0) { txAll1 = txAll0 + 1; }

    /* 交互层：X = 时间（列方向），Y = 频率（行方向，向上为高频） */
    var win = chartWindow('waterfall', { x0: txAll0, x1: txAll1, y0: fxAll0, y1: fxAll1 }, {
      xCount: T, yCount: F,
      fmtX: fmtTimeTick, unitX: 's', fmtY: fmtHzTick, unitY: 'Hz',
      canvas: cv, redraw: drawWaterfall
    });
    var tx0 = win.x0;
    var tx1 = win.x1;
    var fx0 = win.y0;
    var fx1 = win.y1;
    chartGeom('waterfall', m, pw, ph);

    /* 只把窗口覆盖的子图绘制到画布：时间 → 列，频率 → 行（图像第 0 行是最低频？否：
       构建时 iy = F-1-r，故图像第 0 行对应最高频，即绘图区顶部） */
    ctx.imageSmoothingEnabled = false;
    var colA = (tx0 - txAll0) / (txAll1 - txAll0) * (T - 1);
    var colB = (tx1 - txAll0) / (txAll1 - txAll0) * (T - 1);
    var rowLo = (fx0 - fxAll0) / (fxAll1 - fxAll0) * (F - 1);
    var rowHi = (fx1 - fxAll0) / (fxAll1 - fxAll0) * (F - 1);
    var sw = (colB - colA) + 1;
    var sh = (rowHi - rowLo) + 1;
    var sy = (F - 1) - rowHi;
    if (colA < 0) { colA = 0; }
    if (sy < 0) { sy = 0; }
    if (sw < 1) { sw = 1; }
    if (sh < 1) { sh = 1; }
    if (colA + sw > T) { sw = T - colA; }
    if (sy + sh > F) { sh = F - sy; }
    if (sw > 0 && sh > 0) { ctx.drawImage(off, colA, sy, sw, sh, m.l, m.t, pw, ph); }

    ctx.font = MONO_FONT_SM;
    ctx.lineWidth = 1;
    var i;
    var yt = ticks(fx0, fx1, 5);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (i = 0; i < yt.length; i++) {
      var yy = m.t + ph - (yt[i] - fx0) / (fx1 - fx0) * ph;
      yy = Math.round(yy) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(m.l, yy);
      ctx.lineTo(m.l + pw, yy);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(fmtHzTick(yt[i]), m.l - 6, Math.min(m.t + ph - 5, Math.max(m.t + 6, yy)));
    }
    var xt = ticks(tx0, tx1, 5);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (i = 0; i < xt.length; i++) {
      var xx = Math.round(m.l + (xt[i] - tx0) / (tx1 - tx0) * pw) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(xx, m.t);
      ctx.lineTo(xx, m.t + ph);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(fmtTimeTick(xt[i]), Math.min(m.l + pw - 12, Math.max(m.l + 12, xx)), m.t + ph + 5);
    }

    /* 色标 */
    var cbX = m.l + pw + 12;
    var cbW = 11;
    var g = ctx.createLinearGradient(0, m.t + ph, 0, m.t);
    g.addColorStop(0, rgbCss(cmap(0)));
    g.addColorStop(1, rgbCss(cmap(1)));
    ctx.fillStyle = g;
    ctx.fillRect(cbX, m.t, cbW, ph);
    ctx.strokeStyle = COL.frame;
    ctx.strokeRect(cbX + 0.5, m.t + 0.5, cbW - 1, ph - 1);
    ctx.fillStyle = COL.axisText;
    ctx.font = MONO_FONT_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(maxDb.toFixed(0), cbX + cbW + 4, m.t + 5);
    ctx.fillText(((maxDb + minDb) / 2).toFixed(0), cbX + cbW + 4, m.t + ph / 2);
    ctx.fillText(minDb.toFixed(0), cbX + cbW + 4, m.t + ph - 5);
    ctx.textBaseline = 'top';
    ctx.fillText('dB', cbX + cbW + 4, m.t + 14);

    ctx.fillStyle = COL.axisText2;
    ctx.textAlign = 'left';
    ctx.fillText('频率 (Hz)', m.l + 2, m.t + 2);
    ctx.textAlign = 'right';
    ctx.fillText('时间 (s)', m.l + pw, m.t + ph + 5);

    strokeFrame(ctx, m.l, m.t, pw, ph);
    drawViewBadge(ctx, 'waterfall', m, pw, ph);
  }

  /* ================= 星座图 ================= */
  function drawConstellation() {
    var cv = el.constellationCanvas;
    if (!cv) { return; }
    var emp = el.constellationEmpty;
    var d = state.constellation;
    var ok = d && Array.isArray(d.i) && Array.isArray(d.q) && d.i.length > 0;
    if (!ok) {
      if (emp) {
        emp.textContent = state.pending.constellation ? '加载中…'
          : (state.failed.constellation ? '星座数据加载失败（见顶部错误条）'
            : (state.path ? '暂无星座数据' : '等待数据'));
        show(emp, true);
      }
      if (el.constellationMeta) { el.constellationMeta.textContent = '—'; }
      return;
    }
    if (emp) { show(emp, false); }

    var f = fitCanvas(cv);
    var ctx = f.ctx;
    var m = { l: 40, r: 14, t: 12, b: 20 };
    var pw = f.w - m.l - m.r;
    var ph = f.h - m.t - m.b;
    if (pw < 24 || ph < 24) { return; }

    var n = Math.min(d.i.length, d.q.length);
    var i;
    var maxAbs = 0;
    for (i = 0; i < n; i++) {
      if (isNum(d.i[i]) && Math.abs(d.i[i]) > maxAbs) { maxAbs = Math.abs(d.i[i]); }
      if (isNum(d.q[i]) && Math.abs(d.q[i]) > maxAbs) { maxAbs = Math.abs(d.q[i]); }
    }
    if (!(maxAbs > 0)) { maxAbs = 1; }
    maxAbs = maxAbs * 1.12;

    var cx = m.l + pw / 2;
    var cy = m.t + ph / 2;
    var half = Math.min(pw, ph) / 2 - 4;

    /* 交互层：X/Y 都是 I/Q 数据坐标；未缩放时窗口 = ±maxAbs，渲染与原来完全一致 */
    var win = chartWindow('constellation', { x0: -maxAbs, x1: maxAbs, y0: -maxAbs, y1: maxAbs }, {
      xCount: 0, yCount: 0, fmtX: fmtEst, fmtY: fmtEst,
      canvas: cv, redraw: drawConstellation
    });
    chartGeom('constellation', m, pw, ph);
    var vxc = (win.x0 + win.x1) / 2;
    var vyc = (win.y0 + win.y1) / 2;
    var vhx = (win.x1 - win.x0) / 2;
    var vhy = (win.y1 - win.y0) / 2;
    if (!(vhx > 0)) { vhx = 1; }
    if (!(vhy > 0)) { vhy = 1; }
    function px(v) { return cx + (v - vxc) / vhx * half; }
    function py(v) { return cy - (v - vyc) / vhy * half; }

    ctx.font = MONO_FONT_SM;
    ctx.lineWidth = 1;
    var gxs = ticks(win.x0, win.x1, 4);
    var gys = ticks(win.y0, win.y1, 4);
    var ex0 = (win.x1 - win.x0) * 1e-9;
    var ey0 = (win.y1 - win.y0) * 1e-9;
    for (i = 0; i < gxs.length; i++) {
      ctx.strokeStyle = (Math.abs(gxs[i]) <= ex0) ? COL.gridStrong : COL.grid;
      var gx = Math.round(px(gxs[i])) + 0.5;
      ctx.beginPath();
      ctx.moveTo(gx, m.t);
      ctx.lineTo(gx, m.t + ph);
      ctx.stroke();
    }
    for (i = 0; i < gys.length; i++) {
      ctx.strokeStyle = (Math.abs(gys[i]) <= ey0) ? COL.gridStrong : COL.grid;
      var gy = Math.round(py(gys[i])) + 0.5;
      ctx.beginPath();
      ctx.moveTo(m.l, gy);
      ctx.lineTo(m.l + pw, gy);
      ctx.stroke();
    }
    var oxv = px(0);
    var oyv = py(0);
    var qx = (oxv > m.l + 8 && oxv < m.l + pw - 8) ? oxv + 4 : cx + 4;
    var iy = (oyv > m.t + 8 && oyv < m.t + ph - 8) ? oyv + 3 : cy + 3;
    ctx.fillStyle = COL.axisText;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('I', m.l + pw - 10, iy);
    ctx.fillText('Q', qx, m.t + 2);

    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, pw, ph);
    ctx.clip();
    ctx.fillStyle = COL.iq;
    for (i = 0; i < n; i++) {
      if (!isNum(d.i[i]) || !isNum(d.q[i])) { continue; }
      ctx.beginPath();
      ctx.arc(px(d.i[i]), py(d.q[i]), 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    strokeFrame(ctx, m.l, m.t, pw, ph);
    drawViewBadge(ctx, 'constellation', m, pw, ph);

    if (el.constellationMeta) {
      var src = d.source === 'provided' ? '真值参数' : '估计参数';
      el.constellationMeta.textContent = String(d.count) + ' 符号  ·  EVM '
        + (isNum(d.evm_percent) ? d.evm_percent.toFixed(2) + '%' : '—')
        + '  ·  ' + src + '  ·  Rs ' + (isNum(d.symbol_rate) ? d.symbol_rate.toFixed(2) : '—')
        + ' Hz  ·  CFO ' + (isNum(d.cfo_hz) ? d.cfo_hz.toFixed(2) : '—') + ' Hz';
    }
  }

  /* ================= 参数估计 ================= */
  function comparisonFor(a, key) {
    var c = (a && a.comparison) ? a.comparison : null;
    if (!c) { return null; }
    var est = a[key];
    var estV = (est && isNum(est.value)) ? est.value : null;
    var unit = (est && est.unit) ? String(est.unit) : '';
    if (key === 'cfo') {
      var tv = isNum(c.cfo_true_hz) ? c.cfo_true_hz
        : (isNum(c.cfo_error_hz) && estV !== null ? estV - c.cfo_error_hz : null);
      var ev = isNum(c.cfo_error_hz) ? c.cfo_error_hz
        : (isNum(tv) && estV !== null ? estV - tv : null);
      var rv = (isNum(ev) && isNum(tv) && tv !== 0) ? ev / tv : null;
      return { trueV: tv, err: ev, rel: rv, unit: unit || 'Hz' };
    }
    if (key === 'symbol_rate') {
      var tv2 = isNum(c.symbol_rate_true_hz) ? c.symbol_rate_true_hz
        : (isNum(c.symbol_rate_error_hz) && estV !== null ? estV - c.symbol_rate_error_hz : null);
      var ev2 = isNum(c.symbol_rate_error_hz) ? c.symbol_rate_error_hz
        : (isNum(tv2) && estV !== null ? estV - tv2 : null);
      var rv2 = isNum(c.symbol_rate_rel_error) ? c.symbol_rate_rel_error
        : ((isNum(ev2) && isNum(tv2) && tv2 !== 0) ? ev2 / tv2 : null);
      return { trueV: tv2, err: ev2, rel: rv2, unit: unit || 'Hz' };
    }
    if (key === 'snr') {
      var tv3 = isNum(c.es_n0_true_db) ? c.es_n0_true_db
        : (isNum(c.es_n0_error_db) && estV !== null ? estV - c.es_n0_error_db : null);
      var ev3 = isNum(c.es_n0_error_db) ? c.es_n0_error_db
        : (isNum(tv3) && estV !== null ? estV - tv3 : null);
      return { trueV: tv3, err: ev3, rel: null, unit: unit || 'dB' };
    }
    return null;
  }

  function signed(v, d) {
    if (!isNum(v)) { return '—'; }
    var s = v.toFixed(d);
    if (v > 0) { s = '+' + s; }
    return s;
  }

  function renderEstimate() {
    var host = el.estCards;
    if (!host) { return; }
    clear(host);
    var a = state.analyze;
    if (!a) {
      var msg = state.pending.analyze ? '加载中…'
        : (state.failed.analyze ? '分析接口加载失败（见顶部错误条）'
          : (state.path ? '暂无分析结果' : '等待数据：请在左侧选择案例或输入路径'));
      host.appendChild(mk('div', 'empty', msg));
      renderCompare();
      renderAnalyzeMeta();
      return;
    }
    var defs = [
      { key: 'cfo', title: '载频 CFO', hint: '相对基带的载波频偏' },
      { key: 'symbol_rate', title: '符号速率', hint: '每秒符号数' },
      { key: 'snr', title: 'SNR (Es/N0)', hint: '每符号信噪比' }
    ];
    for (var i = 0; i < defs.length; i++) { host.appendChild(estCard(defs[i], a)); }
    renderCompare();
    renderAnalyzeMeta();
  }

  function estCard(def, a) {
    var est = a[def.key];
    var card = mk('div', 'est-card');
    var head = mk('div', 'est-head');
    head.appendChild(mk('span', 'est-title', def.title));
    head.appendChild(mk('span', 'badge badge-' + confClass(est && est.confidence), confText(est && est.confidence)));
    card.appendChild(head);

    var hasVal = !!(est && isNum(est.value));
    if (!hasVal) { card.className += ' is-na'; }
    var val = mk('div', 'est-value');
    val.appendChild(mk('span', 'est-num', hasVal ? fmtEst(est.value) : '—'));
    val.appendChild(mk('span', 'est-unit', (est && est.unit) ? String(est.unit) : ''));
    card.appendChild(val);

    var mth = mk('div', 'est-method');
    mth.textContent = '方法：' + ((est && est.method) ? String(est.method) : '未提供');
    mth.title = (est && est.method) ? String(est.method) : '';
    card.appendChild(mth);

    card.appendChild(mk('div', 'est-method', '说明：' + def.hint));

    var cmp = comparisonFor(a, def.key);
    if (cmp) {
      var line = mk('div', 'est-cmp');
      var errAbs = isNum(cmp.err) ? Math.abs(cmp.err) : null;
      line.appendChild(mk('span', null, '真值 ' + (isNum(cmp.trueV) ? fmtEst(cmp.trueV) + ' ' + cmp.unit : '—')));
      var e1 = mk('span', isNum(cmp.err) ? (errAbs !== null && errAbs < 1e-9 ? 'err-good' : 'err-ok') : 'na');
      e1.textContent = '误差 ' + (isNum(cmp.err) ? signed(cmp.err, def.key === 'snr' ? 3 : 4) + ' ' + cmp.unit : '—');
      line.appendChild(e1);
      if (isNum(cmp.rel)) {
        line.appendChild(mk('span', 'err-ok', '相对 ' + signed(cmp.rel * 100, 3) + '%'));
      }
      card.appendChild(line);
    } else {
      card.appendChild(mk('div', 'est-cmp', '无真值：不做对账'));
    }

    if (est && hasKeys(est.evidence)) {
      card.appendChild(evidenceBlock('证据 evidence', est.evidence));
    }
    return card;
  }

  function renderCompare() {
    var host = el.compareBody;
    if (!host) { return; }
    clear(host);
    var a = state.analyze;
    var c = (a && a.comparison) ? a.comparison : null;
    setText('compareMod', (c && c.modulation) ? String(c.modulation) : (a ? '无真值' : '—'));
    if (!a) {
      host.appendChild(mk('div', 'empty', state.pending.analyze ? '加载中…' : '暂无分析结果'));
      return;
    }
    if (!c || !hasKeys(c)) {
      host.appendChild(mk('div', 'empty', '该案例不含真值（truth），无法对账：仅显示估计值与置信度'));
      return;
    }
    var rows = [
      { k: '载频 CFO', key: 'cfo' },
      { k: '符号速率', key: 'symbol_rate' },
      { k: 'SNR (Es/N0)', key: 'snr' }
    ];
    var tbl = mk('table', 'cmp-table');
    var thead = mk('thead');
    var tr = mk('tr');
    var hs = ['估计量', '估计值', '真值', '绝对误差', '相对误差'];
    for (var i = 0; i < hs.length; i++) { tr.appendChild(mk('th', null, hs[i])); }
    thead.appendChild(tr);
    tbl.appendChild(thead);
    var tb = mk('tbody');
    for (i = 0; i < rows.length; i++) {
      var est = a[rows[i].key];
      var cmp = comparisonFor(a, rows[i].key);
      var r = mk('tr');
      r.appendChild(mk('td', 'k', rows[i].k));
      r.appendChild(mk('td', 'v-est', (est && isNum(est.value)) ? (fmtEst(est.value) + ' ' + (est.unit || '')) : '—'));
      r.appendChild(mk('td', 'v-true', cmp && isNum(cmp.trueV) ? (fmtEst(cmp.trueV) + ' ' + cmp.unit) : '—'));
      var errTd = mk('td', 'v-err', cmp && isNum(cmp.err) ? (signed(cmp.err, rows[i].key === 'snr' ? 3 : 4) + ' ' + cmp.unit) : '—');
      r.appendChild(errTd);
      r.appendChild(mk('td', null, cmp && isNum(cmp.rel) ? (signed(cmp.rel * 100, 3) + '%') : '—'));
      tb.appendChild(r);
    }
    tbl.appendChild(tb);
    host.appendChild(tbl);

    var extra = mk('div', 'kv');
    extra.style.marginTop = '8px';
    if (isNum(c.sps_true) || isNum(c.sps_est)) {
      extra.appendChild(mk('dt', null, 'sps'));
      extra.appendChild(mk('dd', null, '真值 ' + (isNum(c.sps_true) ? String(c.sps_true) : '—')
        + ' / 估计 ' + (isNum(c.sps_est) ? c.sps_est.toFixed(3) : '—')));
    }
    if (isNum(a.sps)) {
      extra.appendChild(mk('dt', null, 'sps (analyze)'));
      extra.appendChild(mk('dd', null, a.sps.toFixed(3)));
    }
    host.appendChild(extra);
  }

  function renderAnalyzeMeta() {
    var host = el.analyzeMeta;
    if (!host) { return; }
    clear(host);
    var a = state.analyze;
    if (!a) { host.appendChild(mk('dt', null, '状态')); host.appendChild(mk('dd', null, state.pending.analyze ? '加载中…' : '无数据')); return; }
    var rows = [
      ['sample_rate', isNum(a.sample_rate) ? a.sample_rate.toFixed(3) + ' Hz' : '—'],
      ['num_samples', isNum(a.num_samples) ? fmtInt(a.num_samples) : '—'],
      ['duration_s', isNum(a.duration_s) ? a.duration_s.toFixed(6) + ' s' : '—'],
      ['noise_floor_psd', isNum(a.noise_floor_psd) ? a.noise_floor_psd.toExponential(4) : '—'],
      ['total_power', isNum(a.total_power) ? a.total_power.toExponential(4) : '—'],
      ['peak.frequency_hz', (a.peak && isNum(a.peak.frequency_hz)) ? a.peak.frequency_hz.toFixed(3) + ' Hz' : '—'],
      ['peak.snr_db', (a.peak && isNum(a.peak.snr_db)) ? a.peak.snr_db.toFixed(3) + ' dB' : '—'],
      ['occupied.bandwidth', (a.occupied && isNum(a.occupied.bandwidth)) ? a.occupied.bandwidth.toFixed(3) + ' Hz' : '—'],
      ['occupied.center', (a.occupied && isNum(a.occupied.center)) ? a.occupied.center.toFixed(3) + ' Hz' : '—'],
      ['occupied.captured_fraction', (a.occupied && isNum(a.occupied.captured_fraction)) ? a.occupied.captured_fraction.toFixed(4) : '—'],
      ['sps', isNum(a.sps) ? a.sps.toFixed(4) : '—']
    ];
    var meta = a.meta || {};
    for (var k in meta) {
      if (Object.prototype.hasOwnProperty.call(meta, k)) {
        rows.push(['meta.' + k, (typeof meta[k] === 'number') ? String(meta[k]) : String(meta[k])]);
      }
    }
    for (var i = 0; i < rows.length; i++) {
      host.appendChild(mk('dt', null, rows[i][0]));
      host.appendChild(mk('dd', null, rows[i][1]));
    }
  }

  /* ================= 特征 ================= */
  function featureGroupsOf(d) {
    var src = (d && d.groups && typeof d.groups === 'object') ? d.groups : FALLBACK_FEATURE_GROUPS;
    var keys = [];
    for (var k in src) {
      if (Object.prototype.hasOwnProperty.call(src, k)) { keys.push(k); }
    }
    if (!keys.length) {
      src = FALLBACK_FEATURE_GROUPS;
      for (k in src) { if (Object.prototype.hasOwnProperty.call(src, k)) { keys.push(k); } }
    }
    keys.sort(function (a, b) {
      var ia = GROUP_ORDER.indexOf(a);
      var ib = GROUP_ORDER.indexOf(b);
      if (ia < 0) { ia = 99; }
      if (ib < 0) { ib = 99; }
      if (ia !== ib) { return ia - ib; }
      return a < b ? -1 : (a > b ? 1 : 0);
    });
    var out = [];
    for (var i = 0; i < keys.length; i++) {
      var v = src[keys[i]];
      if (Array.isArray(v)) { out.push({ key: keys[i], idx: v }); }
      else if (v && typeof v === 'object' && Array.isArray(v.index)) { out.push({ key: keys[i], idx: v.index }); }
    }
    return out;
  }

  function renderFeatures() {
    var host = el.featureGroups;
    if (!host) { return; }
    clear(host);
    renderFeatureSummary();
    var d = state.features;
    if (!d) {
      var msg = state.pending.features ? '加载中…'
        : (state.failed.features ? '特征接口加载失败（见顶部错误条）'
          : (state.path ? '暂无特征数据' : '等待数据：请在左侧选择案例或输入路径'));
      host.appendChild(mk('div', 'empty', msg));
      return;
    }
    var names = (Array.isArray(d.names) && d.names.length) ? d.names : FALLBACK_FEATURE_NAMES;
    var values = Array.isArray(d.values) ? d.values : [];
    if (!values.length && d.values_by_name && typeof d.values_by_name === 'object') {
      values = [];
      for (var i = 0; i < names.length; i++) { values.push(d.values_by_name[names[i]]); }
    }
    var groups = featureGroupsOf(d);
    if (!groups.length) {
      var all = [];
      for (i = 0; i < names.length; i++) { all.push(i); }
      groups = [{ key: 'all', idx: all }];
    }
    for (i = 0; i < groups.length; i++) {
      host.appendChild(featureGroupCard(groups[i], names, values));
    }
    if (!values.length) {
      host.appendChild(mk('div', 'empty', '特征接口返回了 names 但没有 values'));
    }
  }

  function featureGroupCard(g, names, values) {
    var card = mk('div', 'feat-group');
    var head = mk('div', 'feat-group-head');
    head.appendChild(mk('span', 'feat-group-name', GROUP_LABELS[g.key] || String(g.key)));
    head.appendChild(mk('span', 'chip', String(g.idx.length) + ' 维'));
    card.appendChild(head);

    var list = mk('div', 'feat-list');
    var maxAbs = 0;
    var i;
    for (i = 0; i < g.idx.length; i++) {
      var v0 = values[g.idx[i]];
      if (isNum(v0) && Math.abs(v0) > maxAbs) { maxAbs = Math.abs(v0); }
    }
    for (i = 0; i < g.idx.length; i++) {
      var idx = g.idx[i];
      var v = values[idx];
      var row = mk('div', 'feat-row');
      if (!isNum(v)) { row.className += ' is-na'; }
      var nm = mk('span', 'feat-name', names[idx] !== undefined ? String(names[idx]) : ('f' + idx));
      nm.title = nm.textContent + '  (index ' + idx + ')';
      row.appendChild(nm);
      row.appendChild(mk('span', 'feat-val', fmtFeature(v)));
      var track = mk('div', 'bar-track');
      var fill = mk('span', 'bar-fill');
      var w = (isNum(v) && maxAbs > 0) ? Math.abs(v) / maxAbs * 100 : 0;
      fill.style.width = w.toFixed(1) + '%';
      if (isNum(v) && v < 0) { fill.className += ' is-neg'; }
      track.appendChild(fill);
      track.title = names[idx] + ' = ' + fmtFeature(v);
      row.appendChild(track);
      list.appendChild(row);
    }
    card.appendChild(list);
    return card;
  }

  function renderFeatureSummary() {
    var host = el.featureSummary;
    if (!host) { return; }
    clear(host);
    var d = state.features;
    var items = [];
    if (!d) {
      items.push(['维度', state.pending.features ? '加载中…' : '—']);
    } else {
      items.push(['维度', String((Array.isArray(d.names) ? d.names.length : 23)) + ' 维 / ' + String(featureGroupsOf(d).length) + ' 域']);
      items.push(['symbol_rate', isNum(d.symbol_rate) ? d.symbol_rate.toFixed(3) + ' Hz' : '—']);
      items.push(['cfo_hz', isNum(d.cfo_hz) ? d.cfo_hz.toFixed(3) + ' Hz' : '—']);
      items.push(['snr_db', isNum(d.snr_db) ? d.snr_db.toFixed(3) + ' dB' : '—']);
      items.push(['values_by_name', (d.values_by_name && typeof d.values_by_name === 'object') ? '有' : '无']);
    }
    for (var i = 0; i < items.length; i++) {
      var n = mk('span', 'badge');
      n.appendChild(mk('span', null, items[i][0] + ' '));
      n.appendChild(mk('b', null, items[i][1]));
      host.appendChild(n);
    }
  }

  /* ================= 识别 ================= */
  var GATE_TEXT = { ml: '门控 ML 通过', fallback: '门控 物理回退', reject: '门控 拒识' };
  var GATE_CLASS = { ml: 'badge-hi', fallback: 'badge-mid', reject: 'badge-lo' };

  function renderClassify() {
    var d = state.classify;
    var mod = el.verdictMod;
    if (mod) {
      var name = '—';
      var cls = 'verdict-mod mono';
      if (d && d.modulation) {
        name = String(d.modulation).toUpperCase();
        cls = 'verdict-mod mono ' + (d.gate === 'reject' ? 'is-unknown' : (d.gate === 'fallback' ? 'is-fallback' : ''));
      } else if (d && d.gate === 'reject') {
        name = '未识别';
        cls = 'verdict-mod mono is-unknown';
      }
      mod.textContent = name;
      mod.className = cls;
    }
    var sub = el.verdictSub;
    if (sub) {
      if (!d) {
        sub.textContent = state.pending.classify ? '加载中…'
          : (state.failed.classify ? '分类接口加载失败' : (state.path ? '暂无分类结果' : '等待数据'));
      } else {
        sub.textContent = 'gate=' + String(d.gate) + '  ·  阈值 ' + (isNum(d.threshold) ? d.threshold.toFixed(2) : '—')
          + '  ·  ' + (d.calibrated ? '已校准' : '未校准');
      }
    }
    badgeSet('verdictGate', d ? (GATE_TEXT[d.gate] || ('门控 ' + String(d.gate))) : '门控 —', d ? (GATE_CLASS[d.gate] || '') : 'badge-na');
    badgeSet('verdictConf', d && isNum(d.confidence) ? ('置信度 ' + d.confidence.toFixed(3)) : '置信度 —',
      d ? 'badge-' + confClass(d.confidence) : 'badge-na');
    badgeSet('verdictOod', (d && d.ood) ? ('OOD ' + (isNum(d.ood.score) ? d.ood.score.toFixed(2) : '—') + (d.ood.is_outlier === true ? ' 离群' : ' 正常')) : 'OOD —',
      (d && d.ood && d.ood.is_outlier === true) ? 'badge-lo' : (d && d.ood ? 'badge-hi' : 'badge-na'));
    badgeSet('verdictCal', d ? (d.calibrated ? '已校准' : '未校准') : '校准 —', d ? (d.calibrated ? 'badge-accent' : 'badge-na') : 'badge-na');

    renderProbBars();
    renderGateBody();
    renderPhysical();
  }

  function badgeSet(id, text, extra) {
    var n = el[id];
    if (!n) { return; }
    n.textContent = text;
    n.className = 'badge' + (extra ? (' ' + extra) : '');
  }

  function renderProbBars() {
    var host = el.probBars;
    if (!host) { return; }
    clear(host);
    var d = state.classify;
    var probs = (d && d.probabilities && typeof d.probabilities === 'object') ? d.probabilities : null;
    if (!probs || !hasKeys(probs)) {
      setText('probCount', d ? '0' : '—');
      host.appendChild(mk('div', 'empty', !d ? (state.pending.classify ? '加载中…' : '暂无概率数据') : '响应中没有 probabilities'));
      return;
    }
    var list = [];
    for (var k in probs) {
      if (Object.prototype.hasOwnProperty.call(probs, k)) { list.push({ k: k, v: probs[k] }); }
    }
    list.sort(function (a, b) { return (isNum(b.v) ? b.v : -1) - (isNum(a.v) ? a.v : -1); });
    setText('probCount', String(list.length) + ' 类');
    var box = mk('div', 'prob-list');
    for (var i = 0; i < list.length; i++) {
      var row = mk('div', 'prob-row');
      if (i === 0) { row.className += ' is-top'; }
      row.appendChild(mk('span', 'prob-name', String(list[i].k).toUpperCase()));
      var bar = mk('div', 'prob-bar');
      var fill = mk('span');
      var pct = isNum(list[i].v) ? Math.max(0, Math.min(1, list[i].v)) : 0;
      fill.style.width = (pct * 100).toFixed(2) + '%';
      bar.appendChild(fill);
      row.appendChild(bar);
      row.appendChild(mk('span', 'prob-val', isNum(list[i].v) ? list[i].v.toFixed(4) : '—'));
      box.appendChild(row);
    }
    host.appendChild(box);
    if (isNum(d.threshold)) {
      var note = mk('div', 'est-method');
      note.style.marginTop = '8px';
      note.textContent = '门控阈值 ' + d.threshold.toFixed(3)
        + '：最大类概率低于阈值时降级为物理回退或拒识。';
      host.appendChild(note);
    }
  }

  function renderGateBody() {
    var host = el.gateBody;
    if (!host) { return; }
    clear(host);
    var d = state.classify;
    if (!d) {
      host.appendChild(mk('dt', null, '状态'));
      host.appendChild(mk('dd', null, state.pending.classify ? '加载中…' : '无数据'));
      return;
    }
    var rows = [
      ['gate', String(d.gate)],
      ['threshold', isNum(d.threshold) ? d.threshold.toFixed(4) : '—'],
      ['confidence', isNum(d.confidence) ? d.confidence.toFixed(4) : '—'],
      ['calibrated', d.calibrated === true ? 'true' : (d.calibrated === false ? 'false' : '—')],
      ['ood.method', (d.ood && d.ood.method) ? String(d.ood.method) : '—'],
      ['ood.score', (d.ood && isNum(d.ood.score)) ? d.ood.score.toFixed(4) : '—'],
      ['ood.is_outlier', (d.ood && d.ood.is_outlier !== undefined) ? String(d.ood.is_outlier) : '—']
    ];
    var f = d.features || {};
    for (var k in f) {
      if (Object.prototype.hasOwnProperty.call(f, k)) { rows.push(['features.' + k, isNum(f[k]) ? f[k].toFixed(4) : String(f[k])]); }
    }
    for (var i = 0; i < rows.length; i++) {
      host.appendChild(mk('dt', null, rows[i][0]));
      host.appendChild(mk('dd', null, rows[i][1]));
    }
  }

  function renderPhysical() {
    var host = el.physicalBody;
    if (!host) { return; }
    clear(host);
    var d = state.classify;
    if (!d) {
      setText('physicalState', '—');
      host.appendChild(mk('div', 'empty', state.pending.classify ? '加载中…' : '无数据'));
      return;
    }
    var p = d.physical;
    if (!p) {
      setText('physicalState', d.gate === 'fallback' ? '回退失败' : '未触发');
      host.appendChild(mk('div', 'empty', d.gate === 'fallback'
        ? '门控要求物理回退，但响应中 physical 为 null：没有任何物理规则命中该特征组合'
        : '本次未触发物理规则回退（gate=' + String(d.gate) + '）'));
      return;
    }
    setText('physicalState', '已回退');
    var kv = mk('dl', 'kv');
    kv.appendChild(mk('dt', null, 'physical.modulation'));
    kv.appendChild(mk('dd', null, p.modulation ? String(p.modulation).toUpperCase() : '—'));
    kv.appendChild(mk('dt', null, 'physical.reason'));
    kv.appendChild(mk('dd', null, p.reason ? String(p.reason) : '—'));
    host.appendChild(kv);
    if (hasKeys(p)) { host.appendChild(evidenceBlock('physical (raw)', p)); }
  }

  /* ================= 解调（M3） =================
   * 只依赖 /api/demod（契约第 10 节），字段名严格对齐：
   * path / modulation / modulation_source / symbol_rate / cfo_hz / lock / evm_percent /
   * symbols{count,i,q} / traces{index,timing_error,phase_error_rad,freq_estimate_hz} /
   * bits{count,preview,head} / ber|null / llr / warnings
   */

  var LOCK_WORD = { hi: '锁定', mid: '临界', lo: '失锁', na: '—' };

  /* 锁定判据：lock.timing / lock.carrier 是硬判据；metric(0~1) 是软判据。
     locked=false 或 metric < 0.4 → 失锁(红)；metric < 0.7 → 临界(黄)；否则锁定(绿)。 */
  function lockClass(locked, metric) {
    if (locked === false) { return 'lo'; }
    if (isNum(metric)) {
      if (metric < 0.4) { return 'lo'; }
      if (metric < 0.7) { return 'mid'; }
      return 'hi';
    }
    if (locked === true) { return 'hi'; }
    return 'na';
  }

  function lockBadgeText(name, locked, metric, pending, failed) {
    if (locked === undefined && !isNum(metric)) {
      return name + (pending ? ' 加载中' : (failed ? ' 加载失败' : ' —'));
    }
    return name + ' ' + LOCK_WORD[lockClass(locked, metric)];
  }

  /* 按 modulation 解析判决区域类型：MPSK=扇区，MQAM=矩形网格，其余（含 FSK/未知）=不叠加 */
  function demodModKind(mod) {
    var s = String(mod === null || mod === undefined ? '' : mod).toLowerCase().replace(/[\s_\-]/g, '');
    if (s === 'psk' || s === 'bpsk') { return { type: 'psk', order: 2, name: s }; }
    if (s === 'qpsk') { return { type: 'psk', order: 4, name: s }; }
    var m = s.match(/^([0-9]+)psk$/);
    if (m) {
      var po = parseInt(m[1], 10);
      if (po >= 2 && po <= 64 && (po & (po - 1)) === 0) { return { type: 'psk', order: po, name: s }; }
      return null;
    }
    if (s === 'qam') { return { type: 'qam', order: 16, name: s }; }
    m = s.match(/^([0-9]+)qam$/);
    if (m) {
      var qo = parseInt(m[1], 10);
      var lv = Math.round(Math.sqrt(qo));
      if (lv >= 2 && lv <= 32 && lv * lv === qo) { return { type: 'qam', order: qo, name: s }; }
      return null;
    }
    return null;
  }

  function demodModKnown(mod) {
    if (!mod) { return false; }
    var s = String(mod).toLowerCase();
    return DEMOD_MODS.indexOf(s) >= 0 || demodModKind(s) !== null;
  }

  function sci(v) {
    if (!isNum(v)) { return '—'; }
    if (v === 0) { return '0.000e+0'; }
    return v.toExponential(3);
  }

  function legendItem(swatchCls, label) {
    var n = mk('span', 'legend-item');
    n.appendChild(mk('span', swatchCls));
    n.appendChild(mk('span', null, label));
    return n;
  }

  function renderDemod() {
    renderDemodLock();
    renderDemodBits();
    renderDemodBer();
    renderDemodWarnings();
    renderDemodLegends();
  }

  function renderDemodLock() {
    var d = state.demod;
    var pending = !!state.pending.demod;
    var failed = !!state.failed.demod;
    var lk = (d && d.lock && typeof d.lock === 'object') ? d.lock : null;

    var tCls = lk ? lockClass(lk.timing, lk.timing_metric) : 'na';
    var cCls = lk ? lockClass(lk.carrier, lk.carrier_metric) : 'na';
    badgeSet('lockTiming',
      lk ? lockBadgeText('定时锁定', lk.timing, lk.timing_metric) : lockBadgeText('定时锁定', undefined, undefined, pending, failed),
      'badge-' + tCls);
    badgeSet('lockCarrier',
      lk ? lockBadgeText('载波锁定', lk.carrier, lk.carrier_metric) : lockBadgeText('载波锁定', undefined, undefined, pending, failed),
      'badge-' + cCls);

    var mod = (d && d.modulation) ? String(d.modulation) : '';
    if (mod) { badgeSet('lockMod', mod.toUpperCase(), demodModKnown(mod) ? 'badge-accent' : 'badge-mid'); }
    else { badgeSet('lockMod', '调制 —', 'badge-na'); }

    var src = (d && d.modulation_source) ? String(d.modulation_source) : '';
    if (src) { badgeSet('lockSrc', DEMOD_SOURCE_TEXT[src] || ('来源 ' + src), ''); }
    else { badgeSet('lockSrc', '来源 —', 'badge-na'); }

    var metrics = [
      ['lockTimingMetric', lk && isNum(lk.timing_metric) ? lk.timing_metric.toFixed(3) : null],
      ['lockCarrierMetric', lk && isNum(lk.carrier_metric) ? lk.carrier_metric.toFixed(3) : null],
      ['lockSnr', lk && isNum(lk.snr_db) ? (lk.snr_db.toFixed(2) + ' dB') : null],
      ['lockSyms', lk && isNum(lk.locked_symbols) ? fmtInt(lk.locked_symbols) : null],
      ['lockEvm', d && isNum(d.evm_percent) ? (d.evm_percent.toFixed(2) + '%') : null]
    ];
    for (var i = 0; i < metrics.length; i++) {
      var node = el[metrics[i][0]];
      if (!node) { continue; }
      if (metrics[i][1] === null) {
        node.textContent = '—';
        node.className = 'lm-v mono is-na';
      } else {
        node.textContent = metrics[i][1];
        node.className = 'lm-v mono';
      }
    }
  }

  /* 每 8 位一组；只保留 0/1，兼容 preview 里带空格或换行的情况 */
  function groupBits(s) {
    var t = String(s === null || s === undefined ? '' : s).replace(/[^01]/g, '');
    var parts = [];
    for (var i = 0; i < t.length; i += 8) { parts.push(t.slice(i, i + 8)); }
    return parts.join(' ');
  }

  function renderDemodBits() {
    var host = el.bitsBody;
    if (!host) { return; }
    clear(host);
    var d = state.demod;
    if (!d) {
      setText('bitsCount', '—');
      host.appendChild(mk('div', 'empty', state.pending.demod ? '加载中…'
        : (state.failed.demod ? '解调接口加载失败（见顶部错误条）'
          : (state.path ? '暂无解调数据' : '等待数据：请在左侧选择案例或输入路径'))));
      return;
    }
    var b = d.bits;
    if (!b || typeof b !== 'object') {
      setText('bitsCount', '—');
      host.appendChild(mk('div', 'empty', '响应中没有 bits 字段'));
      return;
    }
    setText('bitsCount', isNum(b.count) ? (fmtInt(b.count) + ' bit') : '—');

    var preview = (typeof b.preview === 'string') ? b.preview : '';
    var head = Array.isArray(b.head) ? b.head.join('') : '';
    if (!preview) { preview = head; }
    /* 契约只要求"一段预览"，这里再钳一次，避免后端给超长字符串时把 DOM 撑爆 */
    var BITS_CAP = 1024;
    var shownBits = preview.length > BITS_CAP ? preview.slice(0, BITS_CAP) : preview;
    var pre = mk('pre', 'bits-preview', groupBits(shownBits) || '（无 bits.preview / bits.head）');
    pre.title = '每 8 位一组；显示 ' + String(shownBits.length) + ' / ' + String(preview.length) + ' 位';
    host.appendChild(pre);

    var note = ['preview 位数 ' + String(preview.length)];
    if (preview.length > BITS_CAP) { note.push('仅渲染前 ' + String(BITS_CAP) + ' 位'); }
    if (head) { note.push('head 长度 ' + String(head.length)); }
    host.appendChild(mk('div', 'bits-note', note.join('  ·  ')));

    var llr = d.llr;
    var sec = mk('div', 'ber-sec');
    sec.appendChild(mk('div', 'ber-sec-k', '软判决 LLR'));
    if (!llr || typeof llr !== 'object') {
      sec.appendChild(mk('div', 'empty', '无软判决 LLR（响应中 llr 为空或缺失）'));
      host.appendChild(sec);
      return;
    }
    var kv = mk('dl', 'kv');
    kv.appendChild(mk('dt', null, 'count'));
    kv.appendChild(mk('dd', null, isNum(llr.count) ? fmtInt(llr.count) : '—'));
    kv.appendChild(mk('dt', null, 'mean_abs'));
    kv.appendChild(mk('dd', null, isNum(llr.mean_abs) ? llr.mean_abs.toFixed(3) : '—'));
    kv.appendChild(mk('dt', null, 'min'));
    kv.appendChild(mk('dd', null, isNum(llr.min) ? llr.min.toFixed(3) : '—'));
    kv.appendChild(mk('dt', null, 'max'));
    kv.appendChild(mk('dd', null, isNum(llr.max) ? llr.max.toFixed(3) : '—'));
    sec.appendChild(kv);
    sec.appendChild(llrBar(llr.sample));
    host.appendChild(sec);
  }

  /* LLR 采样条：以中线为零，正=绿（向上）、负=红（向下），长度按 |LLR|/max|LLR| */
  function llrBar(sample) {
    var wrap = mk('div');
    var bar = mk('div', 'llr-bar');
    var vals = Array.isArray(sample) ? sample : [];
    var n = 0;
    var maxAbs = 0;
    var i;
    for (i = 0; i < vals.length; i++) {
      if (!isNum(vals[i])) { continue; }
      n++;
      if (Math.abs(vals[i]) > maxAbs) { maxAbs = Math.abs(vals[i]); }
    }
    if (!n || maxAbs <= 0) {
      bar.appendChild(mk('span', 'bits-note', '（无 llr.sample）'));
      wrap.appendChild(bar);
      return wrap;
    }
    /* 契约说"一小段采样"，这里再钳一次，避免 sample 很长时创建上千个 DOM 节点 */
    var LLR_CAP = 200;
    var shown = Math.min(vals.length, LLR_CAP);
    for (i = 0; i < shown; i++) {
      if (!isNum(vals[i])) { continue; }
      var h = Math.max(1, Math.round(Math.abs(vals[i]) / maxAbs * 26));
      var mark = mk('i', vals[i] >= 0 ? 'llr-pos' : 'llr-neg');
      mark.style.left = (i / shown * 100).toFixed(3) + '%';
      mark.style.top = (vals[i] >= 0 ? (28 - h) : 28) + 'px';
      mark.style.height = h + 'px';
      mark.title = '#' + i + '  ' + vals[i].toFixed(3);
      bar.appendChild(mark);
    }
    wrap.appendChild(bar);
    var cap = mk('div', 'llr-cap');
    cap.appendChild(mk('span', null, '负 ' + (-maxAbs).toFixed(2)));
    cap.appendChild(mk('span', null, '0'));
    cap.appendChild(mk('span', null, '正 ' + maxAbs.toFixed(2)));
    wrap.appendChild(cap);
    if (vals.length > shown) {
      wrap.appendChild(mk('div', 'bits-note', '仅渲染前 ' + String(shown) + ' / ' + String(vals.length) + ' 个采样'));
    }
    return wrap;
  }

  function renderDemodBer() {
    var host = el.berBody;
    if (!host) { return; }
    clear(host);
    var d = state.demod;
    if (!d) {
      setText('berState', '—');
      host.appendChild(mk('div', 'empty', state.pending.demod ? '加载中…'
        : (state.failed.demod ? '解调接口加载失败（见顶部错误条）' : '暂无解调数据')));
      return;
    }
    var b = d.ber;
    if (b === null || b === undefined) {
      setText('berState', '无真值');
      host.appendChild(mk('div', 'empty', '无真值，无法计算 BER'));
      return;
    }
    setText('berState', '有真值');
    var kv = mk('dl', 'kv');
    function row(k, v, cls) {
      kv.appendChild(mk('dt', null, k));
      var dd = mk('dd', null, v);
      if (cls) { dd.className = cls; }
      kv.appendChild(dd);
    }
    row('bit_errors 误码数', isNum(b.bit_errors) ? fmtInt(b.bit_errors) : '—');
    row('compared_bits 比较位数', isNum(b.compared_bits) ? fmtInt(b.compared_bits) : '—');
    var berBad = isNum(b.bit_error_rate) && b.bit_error_rate > 0;
    row('bit_error_rate 误码率', isNum(b.bit_error_rate) ? sci(b.bit_error_rate) : '—', berBad ? 'ber-bad' : '');
    row('symbol_error_rate 符号错误率',
      isNum(b.symbol_error_rate) ? (sci(b.symbol_error_rate) + '（' + (b.symbol_error_rate * 100).toFixed(3) + '%）') : '—');
    host.appendChild(kv);
  }

  function renderDemodWarnings() {
    var host = el.demodWarnings;
    if (!host) { return; }
    clear(host);
    var d = state.demod;
    if (!d) {
      setText('demodWarnCount', '—');
      host.appendChild(mk('div', 'empty', state.pending.demod ? '加载中…'
        : (state.failed.demod ? '解调接口加载失败（见顶部错误条）' : '暂无解调数据')));
      return;
    }
    var w = Array.isArray(d.warnings) ? d.warnings : [];
    setText('demodWarnCount', String(w.length) + ' 条');
    if (!w.length) {
      var ok = mk('div', 'alert-item alert-ok');
      ok.appendChild(mk('span', null, '无解调告警'));
      host.appendChild(ok);
      return;
    }
    var ul = mk('ul', 'warnlist');
    for (var i = 0; i < w.length; i++) {
      var li = mk('li', 'alert-item alert-' + alertLevel(w[i], 'warn'));
      li.appendChild(mk('span', null, String(w[i])));
      ul.appendChild(li);
    }
    host.appendChild(ul);
  }

  function renderDemodLegends() {
    var host = el.demodConstLegend;
    var d = state.demod;
    if (host) {
      clear(host);
      if (d) {
        host.appendChild(legendItem('legend-swatch', '判决前符号'));
        if (state.ui.demodRegions && demodModKind(d.modulation)) {
          host.appendChild(legendItem('legend-swatch is-region', '判决区域'));
        }
      }
    }
    var sh = el.demodSyncLegend;
    if (sh) {
      clear(sh);
      sh.appendChild(legendItem('legend-swatch is-sync-timing', 'timing_error'));
      sh.appendChild(legendItem('legend-swatch is-sync-phase', 'phase_error_rad'));
      sh.appendChild(legendItem('legend-swatch is-sync-freq', 'freq_estimate_hz'));
    }
  }

  /* ---- 星座图（判决前轨迹 + 可选判决区域） ---- */
  function drawDemodConstellation() {
    var cv = el.demodConstellationCanvas;
    if (!cv) { return; }
    var emp = el.demodConstellationEmpty;
    var d = state.demod;
    var sym = (d && d.symbols && typeof d.symbols === 'object') ? d.symbols : null;
    var n = (sym && Array.isArray(sym.i) && Array.isArray(sym.q)) ? Math.min(sym.i.length, sym.q.length) : 0;
    if (!d || !sym || n <= 0) {
      if (emp) {
        emp.textContent = state.pending.demod ? '加载中…'
          : (state.failed.demod ? '解调接口加载失败（见顶部错误条）'
            : (state.path ? '暂无解调数据（响应中没有 symbols）' : '等待数据：请在左侧选择案例或输入路径'));
        show(emp, true);
      }
      if (el.demodConstMeta) { el.demodConstMeta.textContent = '—'; }
      return;
    }
    if (emp) { show(emp, false); }

    var f = fitCanvas(cv);
    var ctx = f.ctx;
    var m = { l: 42, r: 16, t: 14, b: 20 };
    var pw = f.w - m.l - m.r;
    var ph = f.h - m.t - m.b;
    if (pw < 24 || ph < 24) { return; }

    var i;
    var rawMax = 0;
    for (i = 0; i < n; i++) {
      if (isNum(sym.i[i]) && Math.abs(sym.i[i]) > rawMax) { rawMax = Math.abs(sym.i[i]); }
      if (isNum(sym.q[i]) && Math.abs(sym.q[i]) > rawMax) { rawMax = Math.abs(sym.q[i]); }
    }
    if (!(rawMax > 0)) { rawMax = 1; }
    var maxAbs = rawMax * 1.12;
    var cx = m.l + pw / 2;
    var cy = m.t + ph / 2;
    var half = Math.min(pw, ph) / 2 - 6;

    var win = chartWindow('demodConst', { x0: -maxAbs, x1: maxAbs, y0: -maxAbs, y1: maxAbs }, {
      xCount: 0, yCount: 0, fmtX: fmtEst, fmtY: fmtEst,
      canvas: cv, redraw: drawDemodConstellation
    });
    chartGeom('demodConst', m, pw, ph);
    var vxc = (win.x0 + win.x1) / 2;
    var vyc = (win.y0 + win.y1) / 2;
    var vhx = (win.x1 - win.x0) / 2;
    var vhy = (win.y1 - win.y0) / 2;
    if (!(vhx > 0)) { vhx = 1; }
    if (!(vhy > 0)) { vhy = 1; }
    function px(v) { return cx + (v - vxc) / vhx * half; }
    function py(v) { return cy - (v - vyc) / vhy * half; }

    ctx.font = MONO_FONT_SM;
    ctx.lineWidth = 1;
    var gxs = ticks(win.x0, win.x1, 4);
    var gys = ticks(win.y0, win.y1, 4);
    var ex0 = (win.x1 - win.x0) * 1e-9;
    var ey0 = (win.y1 - win.y0) * 1e-9;
    for (i = 0; i < gxs.length; i++) {
      ctx.strokeStyle = (Math.abs(gxs[i]) <= ex0) ? COL.gridStrong : COL.grid;
      var gx = Math.round(px(gxs[i])) + 0.5;
      ctx.beginPath();
      ctx.moveTo(gx, m.t);
      ctx.lineTo(gx, m.t + ph);
      ctx.stroke();
    }
    for (i = 0; i < gys.length; i++) {
      ctx.strokeStyle = (Math.abs(gys[i]) <= ey0) ? COL.gridStrong : COL.grid;
      var gy = Math.round(py(gys[i])) + 0.5;
      ctx.beginPath();
      ctx.moveTo(m.l, gy);
      ctx.lineTo(m.l + pw, gy);
      ctx.stroke();
    }

    var kind = demodModKind(d.modulation);
    if (state.ui.demodRegions && kind) {
      drawDecisionRegions(ctx, kind, px, py, m, pw, ph, rawMax, maxAbs, px(0), py(0), half);
    } else if (!kind) {
      ctx.fillStyle = COL.axisText;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(d.modulation ? ('调制 ' + String(d.modulation) + '：无判决区域模板') : '调制未知：不叠加判决区域', m.l + 4, m.t + 2);
    }

    var oxv = px(0);
    var oyv = py(0);
    var qx2 = (oxv > m.l + 8 && oxv < m.l + pw - 8) ? oxv + 4 : cx + 4;
    var iy2 = (oyv > m.t + 8 && oyv < m.t + ph - 8) ? oyv + 3 : cy + 3;
    ctx.fillStyle = COL.axisText;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('I', m.l + pw - 10, iy2);
    ctx.fillText('Q', qx2, m.t + 2);

    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, pw, ph);
    ctx.clip();
    ctx.fillStyle = COL.iq;
    for (i = 0; i < n; i++) {
      if (!isNum(sym.i[i]) || !isNum(sym.q[i])) { continue; }
      ctx.beginPath();
      ctx.arc(px(sym.i[i]), py(sym.q[i]), 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    strokeFrame(ctx, m.l, m.t, pw, ph);
    drawViewBadge(ctx, 'demodConst', m, pw, ph);

    if (isNum(d.evm_percent)) {
      ctx.font = MONO_FONT;
      ctx.fillStyle = COL.axisText2;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText('EVM ' + d.evm_percent.toFixed(2) + '%', m.l + pw - 4, m.t + 2);
    }

    if (el.demodConstMeta) {
      var symTotal = isNum(sym.count) ? sym.count : n;
      var parts = [fmtInt(n) + ' 符号'];
      if (symTotal !== n) { parts.push('symbols.count ' + fmtInt(symTotal)); }
      parts.push(isNum(d.evm_percent) ? ('EVM ' + d.evm_percent.toFixed(2) + '%') : 'EVM —');
      parts.push(d.modulation ? String(d.modulation).toUpperCase() : '调制 —');
      parts.push(isNum(d.symbol_rate) ? ('Rs ' + d.symbol_rate.toFixed(2) + ' Hz') : 'Rs —');
      el.demodConstMeta.textContent = parts.join('  ·  ');
    }
  }

  /* 判决区域：MPSK 画过原点的扇区边界，MQAM 画矩形判决网格；不支持的类型返回 false */
  function drawDecisionRegions(ctx, kind, px, py, m, pw, ph, rawMax, maxAbs, cx, cy, half) {
    var i;
    if (kind.type === 'psk') {
      var M = kind.order;
      if (M < 2 || M > 64 || (M & (M - 1)) !== 0) { return false; }
      ctx.save();
      ctx.beginPath();
      ctx.rect(m.l, m.t, pw, ph);
      ctx.clip();
      ctx.strokeStyle = COL.regionLine;
      ctx.lineWidth = 1;
      var len = half * 1.5;
      for (i = 0; i < M / 2; i++) {
        var th = Math.PI / M + i * (2 * Math.PI / M);
        var sx = Math.cos(th);
        var sy = -Math.sin(th);
        ctx.beginPath();
        ctx.moveTo(cx - sx * len, cy - sy * len);
        ctx.lineTo(cx + sx * len, cy + sy * len);
        ctx.stroke();
      }
      ctx.restore();
      return true;
    }
    if (kind.type === 'qam') {
      var L = Math.round(Math.sqrt(kind.order));
      if (L < 2 || L > 32 || L * L !== kind.order) { return false; }
      var maxLevel = L - 1;
      var unit = rawMax / maxLevel;
      ctx.save();
      ctx.beginPath();
      ctx.rect(m.l, m.t, pw, ph);
      ctx.clip();
      ctx.strokeStyle = COL.regionLine;
      ctx.lineWidth = 1;
      for (i = 0; i < L - 1; i++) {
        var v = (-(maxLevel - 1) + 2 * i) * unit;
        var gx = Math.round(px(v)) + 0.5;
        var gy = Math.round(py(v)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(gx, m.t);
        ctx.lineTo(gx, m.t + ph);
        ctx.moveTo(m.l, gy);
        ctx.lineTo(m.l + pw, gy);
        ctx.stroke();
      }
      ctx.restore();
      return true;
    }
    return false;
  }

  /* ---- 同步轨迹（本页重点）：三个独立子图共用横轴（符号序号） ---- */
  var SYNC_DEFS = [
    { key: 'timing_error', label: '定时误差', unit: '归一化', color: COL.syncTiming },
    { key: 'phase_error_rad', label: '载波相位误差', unit: 'rad', color: COL.syncPhase },
    { key: 'freq_estimate_hz', label: '载波频率估计', unit: 'Hz', color: COL.syncFreq }
  ];

  function syncTick(v, def) {
    if (def.unit === 'Hz') { return fmtHzTick(v); }
    var a = Math.abs(v);
    if (a >= 100) { return v.toFixed(0); }
    if (a >= 1) { return v.toFixed(2); }
    return v.toFixed(3);
  }

  function drawSyncTraces() {
    var cv = el.demodSyncCanvas;
    if (!cv) { return; }
    var emp = el.demodSyncEmpty;
    var d = state.demod;
    var t = (d && d.traces && typeof d.traces === 'object') ? d.traces : null;
    var idxArr = (t && Array.isArray(t.index)) ? t.index : null;
    var defs = [];
    var i;
    for (i = 0; i < SYNC_DEFS.length; i++) {
      var raw = (t && Array.isArray(t[SYNC_DEFS[i].key])) ? t[SYNC_DEFS[i].key] : null;
      defs.push({ def: SYNC_DEFS[i], vals: raw, count: 0 });
    }
    var hasAny = false;
    for (i = 0; i < defs.length; i++) { if (defs[i].vals && defs[i].vals.length) { hasAny = true; } }

    if (!d || !t || !hasAny) {
      if (emp) {
        emp.textContent = state.pending.demod ? '加载中…'
          : (state.failed.demod ? '解调接口加载失败（见顶部错误条）'
            : (state.path ? '暂无同步轨迹数据（响应中 traces 为空）' : '等待数据：请在左侧选择案例或输入路径'));
        show(emp, true);
      }
      if (el.demodSyncMeta) { el.demodSyncMeta.textContent = '—'; }
      return;
    }
    if (emp) { show(emp, false); }

    /* 长度容错：每条序列各自按 min(index.length, 序列长度) 截断，index 缺失则退化为点序号 */
    var maxLen = 0;
    for (i = 0; i < defs.length; i++) {
      var v = defs[i].vals;
      var lim = 0;
      if (v) {
        lim = idxArr ? Math.min(v.length, idxArr.length) : v.length;
        var last = -1;
        for (var j = 0; j < lim; j++) { if (isNum(v[j])) { last = j; } }
        defs[i].count = last + 1;
      }
      if (defs[i].count > maxLen) { maxLen = defs[i].count; }
    }

    var xmin = 0;
    var xmax = Math.max(1, maxLen - 1);
    if (idxArr) {
      var gotX = false;
      var limX = Math.min(idxArr.length, Math.max(1, maxLen));
      for (i = 0; i < limX; i++) {
        if (!isNum(idxArr[i])) { continue; }
        if (!gotX) { xmin = idxArr[i]; xmax = idxArr[i]; gotX = true; }
        else if (idxArr[i] < xmin) { xmin = idxArr[i]; }
        else if (idxArr[i] > xmax) { xmax = idxArr[i]; }
      }
      if (!gotX) { xmin = 0; xmax = Math.max(1, maxLen - 1); }
    }
    if (xmax <= xmin) { xmax = xmin + 1; }

    /* 交互层：三个子图共用横轴（符号序号），只允许 X 缩放/平移（三条曲线单位不同，Y 缩放无意义） */
    var win = chartWindow('demodSync', { x0: xmin, x1: xmax, y0: 0, y1: 1 }, {
      xCount: maxLen, yCount: 0, yZoom: false,
      fmtX: fmtInt, unitX: '',
      canvas: cv, redraw: drawSyncTraces
    });
    xmin = win.x0;
    xmax = win.x1;

    if (el.demodSyncMeta) {
      var bits = [];
      bits.push('x = 符号序号');
      bits.push('index ' + (idxArr ? String(idxArr.length) : '缺失'));
      for (i = 0; i < defs.length; i++) {
        bits.push(defs[i].def.key + ' ' + (defs[i].vals ? String(defs[i].vals.length) : '缺失'));
      }
      bits.push('绘制 ' + fmtInt(maxLen) + ' 点');
      el.demodSyncMeta.textContent = bits.join('  ·  ');
    }

    var f = fitCanvas(cv);
    var ctx = f.ctx;
    var m = { l: 70, r: 18, t: 12, b: 26 };
    var pw = f.w - m.l - m.r;
    var ph = f.h - m.t - m.b;
    if (pw < 48 || ph < 72) { return; }
    var gap = 18;
    var each = (ph - gap * (defs.length - 1)) / defs.length;
    if (each < 24) { return; }
    chartGeom('demodSync', m, pw, ph);

    function xAt(k) {
      var xv = (idxArr && isNum(idxArr[k])) ? idxArr[k] : k;
      return m.l + (xv - xmin) / (xmax - xmin) * pw;
    }

    ctx.font = MONO_FONT_SM;
    ctx.lineWidth = 1;
    var k;
    var xt = ticks(xmin, xmax, 5);
    var lastBottom = 0;

    for (i = 0; i < defs.length; i++) {
      var s = defs[i];
      var y0 = m.t + i * (each + gap);
      lastBottom = y0 + each;

      for (k = 0; k < xt.length; k++) {
        var gx = Math.round(m.l + (xt[k] - xmin) / (xmax - xmin) * pw) + 0.5;
        ctx.strokeStyle = COL.grid;
        ctx.beginPath();
        ctx.moveTo(gx, y0);
        ctx.lineTo(gx, y0 + each);
        ctx.stroke();
      }

      var mn = Infinity;
      var mx = -Infinity;
      for (k = 0; k < s.count; k++) {
        var nv = s.vals[k];
        if (!isNum(nv)) { continue; }
        if (nv < mn) { mn = nv; }
        if (nv > mx) { mx = nv; }
      }
      if (mn === Infinity) { mn = 0; mx = 1; }
      if (mn > 0) { mn = 0; }
      if (mx < 0) { mx = 0; }
      if (mx - mn < 1e-12) { mx = mn + 1; }
      var pad = (mx - mn) * 0.12;
      mn -= pad;
      mx += pad;

      function yAt(val) { return y0 + each - (val - mn) / (mx - mn) * each; }

      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      var yt = ticks(mn, mx, 2);
      for (k = 0; k < yt.length; k++) {
        var yy = Math.round(yAt(yt[k])) + 0.5;
        ctx.strokeStyle = COL.grid;
        ctx.beginPath();
        ctx.moveTo(m.l, yy);
        ctx.lineTo(m.l + pw, yy);
        ctx.stroke();
        ctx.fillStyle = COL.axisText;
        ctx.fillText(syncTick(yt[k], s.def), m.l - 6, yy);
      }

      if (mn <= 0 && mx >= 0) {
        ctx.save();
        ctx.strokeStyle = COL.gridStrong;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        var zy = Math.round(yAt(0)) + 0.5;
        ctx.moveTo(m.l, zy);
        ctx.lineTo(m.l + pw, zy);
        ctx.stroke();
        ctx.restore();
      }

      ctx.fillStyle = s.def.color;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(s.def.label + '（' + s.def.unit + '）', m.l + 4, y0 + 2);
      ctx.fillStyle = COL.axisText;
      ctx.textAlign = 'right';
      ctx.fillText(s.count ? (fmtInt(s.count) + ' 点') : '无数据', m.l + pw - 4, y0 + 2);

      if (s.count > 0) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(m.l, y0, pw, each);
        ctx.clip();
        ctx.strokeStyle = s.def.color;
        ctx.lineWidth = 1.4;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        var started = false;
        for (k = 0; k < s.count; k++) {
          var pv = s.vals[k];
          if (!isNum(pv)) { started = false; continue; }
          var pxx = xAt(k);
          var pyy = yAt(pv);
          if (!started) { ctx.moveTo(pxx, pyy); started = true; } else { ctx.lineTo(pxx, pyy); }
        }
        ctx.stroke();
        ctx.lineWidth = 1;
        ctx.restore();
      } else {
        ctx.fillStyle = COL.axisText;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('该序列无数据', m.l + pw / 2, y0 + each / 2);
      }

      strokeFrame(ctx, m.l, y0, pw, each);
    }

    ctx.fillStyle = COL.axisText;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var titleW = 60;
    for (k = 0; k < xt.length; k++) {
      var tx = Math.round(m.l + (xt[k] - xmin) / (xmax - xmin) * pw);
      tx = Math.min(m.l + pw - titleW, Math.max(m.l + 10, tx));
      ctx.fillText(fmtInt(xt[k]), tx, lastBottom + 5);
    }
    ctx.fillStyle = COL.axisText2;
    ctx.textAlign = 'right';
    ctx.fillText('符号序号', m.l + pw, lastBottom + 5);
    drawViewBadge(ctx, 'demodSync', { l: m.l, t: m.t }, pw, lastBottom - m.t);
  }

  /* ================= 译码（M4） =================
   * 只依赖 /api/fec（契约 §11），字段名严格对齐：
   * updated_at / report_path / catalogue[]{id,name,params,rate,note} / ebn0_db[] /
   * curves{id}{ber,bler} / metrics{soft_gain_db,soft_gain_note,rs_random_correction,
   * rs_burst_demo,ccsds_interleave_depth,ldpc,checks[]} / notes[]
   * 容错原则：curves 每条曲线逐点判定，长度不等或含 null/0 只跳过该点，绝不整页崩。
   */

  /* 曲线顺序固定按契约目录；curves 里多出来的键追加在后面，保证不丢数据 */
  var FEC_ORDER = ['uncoded', 'conv_hard', 'conv_soft', 'rs', 'ccsds', 'ldpc'];
  /* 曲线颜色是"编码分类"的数据编码色，与图例一一对应 */
  var FEC_COLORS = {
    uncoded: '#8d9aa8', conv_hard: '#e0a23a', conv_soft: '#2ed3b7',
    rs: '#57c98a', ccsds: '#5fa8d3', ldpc: '#b48ee0'
  };
  var FEC_FALLBACK_COLORS = ['#8d9aa8', '#e0a23a', '#2ed3b7', '#57c98a', '#5fa8d3', '#b48ee0'];
  var FEC_YMIN_FLOOR = 1e-6;

  var fecGeom = null;
  var fecHoverIndex = -1;

  function fecColor(id, idx) {
    if (FEC_COLORS[id]) { return FEC_COLORS[id]; }
    return FEC_FALLBACK_COLORS[idx % FEC_FALLBACK_COLORS.length];
  }

  /* 契约顶层就是 catalogue/ebn0_db/curves；若后端把 report 再包一层也兼容 */
  function fecPayload() {
    var d = state.fec;
    if (!d || typeof d !== 'object') { return null; }
    if (d.catalogue || d.curves || d.ebn0_db) { return d; }
    if (d.report && typeof d.report === 'object'
      && (d.report.catalogue || d.report.curves || d.report.ebn0_db)) { return d.report; }
    return d;
  }

  function fecCatalogue(p) {
    var out = [];
    var arr = (p && Array.isArray(p.catalogue)) ? p.catalogue : [];
    for (var i = 0; i < arr.length; i++) {
      if (arr[i] && typeof arr[i] === 'object') { out.push(arr[i]); }
    }
    return out;
  }

  function fecCurves(p) {
    return (p && p.curves && typeof p.curves === 'object') ? p.curves : null;
  }

  function fecCurveIds(p) {
    var cv = fecCurves(p);
    if (!cv) { return []; }
    var out = [];
    var seen = {};
    var k;
    var i;
    for (i = 0; i < FEC_ORDER.length; i++) {
      k = FEC_ORDER[i];
      if (cv[k] && typeof cv[k] === 'object') { out.push(k); seen[k] = true; }
    }
    var cat = fecCatalogue(p);
    for (i = 0; i < cat.length; i++) {
      k = cat[i].id ? String(cat[i].id) : '';
      if (k && !seen[k] && cv[k] && typeof cv[k] === 'object') { out.push(k); seen[k] = true; }
    }
    for (k in cv) {
      if (Object.prototype.hasOwnProperty.call(cv, k) && !seen[k] && cv[k] && typeof cv[k] === 'object') {
        out.push(k); seen[k] = true;
      }
    }
    return out;
  }

  function fecNameMap(p) {
    var m = {};
    var cat = fecCatalogue(p);
    for (var i = 0; i < cat.length; i++) {
      if (cat[i] && cat[i].id) { m[String(cat[i].id)] = cat[i].name ? String(cat[i].name) : String(cat[i].id); }
    }
    return m;
  }

  function fecMetricKey() { return state.ui.fecMetric === 'bler' ? 'bler' : 'ber'; }
  function fecMetricLabel() { return fecMetricKey() === 'bler' ? 'BLER' : 'BER'; }

  function fmtFecVal(v) {
    if (v === null || v === undefined) { return '—'; }
    if (typeof v === 'boolean') { return v ? 'true' : 'false'; }
    if (typeof v === 'number') {
      if (!isFinite(v)) { return '—'; }
      var a = Math.abs(v);
      if (a >= 1e-3) {
        if (Number.isInteger(v)) { return String(v); }
        return String(parseFloat(v.toPrecision(4)));
      }
      return v.toExponential(2);
    }
    return String(v);
  }

  function fmtFecRate(v) {
    var n = (typeof v === 'number') ? v : parseFloat(v);
    if (typeof n !== 'number' || !isFinite(n)) { return '—'; }
    return n.toFixed(2);
  }

  function fmtFecBer(v) {
    if (!isNum(v)) { return '—'; }
    if (v === 0) { return '0'; }
    return v.toExponential(2);
  }

  function fecEmptyText(what) {
    if (state.pending.fec) { return '加载中…'; }
    if (state.failed.fec) { return '译码报告加载失败（见顶部错误条）'; }
    if (!state.fec) { return '尚未加载译码报告：切换到本页或点「重新加载」'; }
    return what || '响应中没有 catalogue / curves';
  }

  function syncSegBtn(node, on) {
    if (!node) { return; }
    node.className = on ? 'seg-btn is-active' : 'seg-btn';
    node.setAttribute('aria-pressed', on ? 'true' : 'false');
  }

  function syncFecMetricButtons() {
    var isBler = fecMetricKey() === 'bler';
    syncSegBtn(el.fecMetricBer, !isBler);
    syncSegBtn(el.fecMetricBler, isBler);
  }

  function setFecMetric(m) {
    state.ui.fecMetric = (m === 'bler') ? 'bler' : 'ber';
    syncFecMetricButtons();
    fecHoverIndex = -1;
    if (el.fecTip) { show(el.fecTip, false); }
    drawFecCurves();
  }

  /* ---- 关键指标卡 ---- */
  function renderFecCards() {
    var host = el.fecCards;
    if (!host) { return; }
    clear(host);
    var p = fecPayload();
    if (!p) {
      host.appendChild(mk('div', 'empty', fecEmptyText()));
      return;
    }
    var m = (p.metrics && typeof p.metrics === 'object') ? p.metrics : null;
    if (!m) {
      host.appendChild(mk('div', 'empty', fecEmptyText('响应中没有 metrics')));
      return;
    }
    var rsc = (m.rs_random_correction && typeof m.rs_random_correction === 'object') ? m.rs_random_correction : null;
    var burst = (m.rs_burst_demo && typeof m.rs_burst_demo === 'object') ? m.rs_burst_demo : null;
    var ldpc = (m.ldpc && typeof m.ldpc === 'object') ? m.ldpc : null;
    var rec = (burst && typeof burst.recovered === 'boolean') ? burst.recovered : null;

    var defs = [
      {
        title: '软判决增益',
        num: isNum(m.soft_gain_db) ? m.soft_gain_db.toFixed(2) : '—',
        unit: isNum(m.soft_gain_db) ? 'dB' : '',
        method: 'soft_gain_db' + (m.soft_gain_note ? (' · ' + String(m.soft_gain_note)) : '')
      },
      {
        title: 'RS 纠错能力 (t)',
        num: (rsc && isNum(rsc.t)) ? fmtInt(rsc.t) : '—',
        unit: (rsc && isNum(rsc.t)) ? '符号' : '',
        method: rsc
          ? ('rs_random_correction · RS(' + fmtInt(rsc.n) + ',' + fmtInt(rsc.k) + ') · 随机纠错 '
            + fmtInt(rsc.corrected_symbols) + ' 符号（t=' + fmtInt(rsc.t) + '）')
          : '响应中没有 rs_random_correction'
      },
      {
        title: 'RS 突发纠错演示',
        num: rec === null ? '—' : (rec ? '已恢复' : '未恢复'),
        unit: '',
        numCls: rec === null ? ' is-na' : (rec ? ' is-ok' : ' is-bad'),
        method: burst
          ? ('rs_burst_demo · 突发长度 '
            + (isNum(burst.burst_length_symbols) ? fmtInt(burst.burst_length_symbols) : '—') + ' 符号')
          : '响应中没有 rs_burst_demo'
      },
      {
        title: 'CCSDS 交织深度',
        num: isNum(m.ccsds_interleave_depth) ? fmtInt(m.ccsds_interleave_depth) : '—',
        unit: isNum(m.ccsds_interleave_depth) ? 'I' : '',
        method: 'ccsds_interleave_depth · 交织深度 I='
          + (isNum(m.ccsds_interleave_depth) ? fmtInt(m.ccsds_interleave_depth) : '—')
      },
      {
        title: 'LDPC 码长 / 码率 / 迭代',
        num: (ldpc && isNum(ldpc.n)) ? fmtInt(ldpc.n) : '—',
        unit: (ldpc && isNum(ldpc.k)) ? ('/ ' + fmtInt(ldpc.k)) : '',
        method: ldpc
          ? ('ldpc · 码率 '
            + ((isNum(ldpc.n) && isNum(ldpc.k) && ldpc.n > 0) ? (ldpc.k / ldpc.n).toFixed(2) : '—')
            + ' · 迭代 ' + (isNum(ldpc.iterations) ? fmtInt(ldpc.iterations) : '—') + ' 次')
          : '响应中没有 ldpc'
      }
    ];

    for (var i = 0; i < defs.length; i++) {
      var d = defs[i];
      var card = mk('div', 'est-card');
      var head = mk('div', 'est-head');
      head.appendChild(mk('span', 'est-title', d.title));
      card.appendChild(head);
      var val = mk('div', 'est-value');
      var numEl = mk('span', 'est-num', d.num);
      if (d.numCls) { numEl.className += d.numCls; }
      if (d.num === '—' && !d.numCls) { numEl.className += ' is-na'; }
      val.appendChild(numEl);
      if (d.unit) { val.appendChild(mk('span', 'est-unit', d.unit)); }
      card.appendChild(val);
      card.appendChild(mk('div', 'est-method', d.method));
      host.appendChild(card);
    }
  }

  /* ---- 编码目录表 ---- */
  function renderFecCatalogue() {
    var host = el.fecCatalogue;
    if (!host) { return; }
    clear(host);
    var p = fecPayload();
    var cat = fecCatalogue(p);
    setText('fecCatCount', p ? (String(cat.length) + ' 条') : '—');
    if (!cat.length) {
      host.appendChild(mk('div', 'empty', p ? '编码目录为空（catalogue 缺失或为空数组）' : fecEmptyText()));
      return;
    }
    var tbl = mk('table', 'fec-table');
    var thead = mk('thead');
    var tr = mk('tr');
    var hs = ['名称', '参数', '码率', '备注'];
    for (var i = 0; i < hs.length; i++) { tr.appendChild(mk('th', null, hs[i])); }
    thead.appendChild(tr);
    tbl.appendChild(thead);
    var tb = mk('tbody');
    for (i = 0; i < cat.length; i++) {
      var c = cat[i];
      var r = mk('tr');
      var nameTd = mk('td', 'k');
      nameTd.appendChild(mk('span', null, c.name ? String(c.name) : (c.id ? String(c.id) : '—')));
      if (c.id) { nameTd.appendChild(mk('span', 'cell-id', String(c.id))); }
      r.appendChild(nameTd);
      r.appendChild(mk('td', 'note', c.params ? String(c.params) : '—'));
      r.appendChild(mk('td', 'rate', fmtFecRate(c.rate)));
      r.appendChild(mk('td', 'note', c.note ? String(c.note) : '—'));
      tb.appendChild(r);
    }
    tbl.appendChild(tb);
    host.appendChild(tbl);
  }

  /* ---- 验收清单 ---- */
  function renderFecChecks() {
    var host = el.fecChecks;
    if (!host) { return; }
    clear(host);
    var p = fecPayload();
    var m = (p && p.metrics && typeof p.metrics === 'object') ? p.metrics : null;
    var arr = (m && Array.isArray(m.checks)) ? m.checks : [];
    var list = [];
    for (var i = 0; i < arr.length; i++) {
      if (arr[i] && typeof arr[i] === 'object') { list.push(arr[i]); }
    }
    var passed = 0;
    var failed = 0;
    for (i = 0; i < list.length; i++) {
      if (list[i].pass === true) { passed++; }
      else if (list[i].pass === false) { failed++; }
    }
    setText('fecCheckCount', p ? (failed ? (String(passed) + ' 通过 / ' + String(failed) + ' 未通过')
      : (String(passed) + ' 通过')) : '—');
    if (!list.length) {
      host.appendChild(mk('div', 'empty', p ? '验收清单为空（metrics.checks 缺失或为空数组）' : fecEmptyText()));
      return;
    }
    var tbl = mk('table', 'fec-table');
    var thead = mk('thead');
    var tr = mk('tr');
    var hs = ['名称', '实测值', '目标值', '结果'];
    for (i = 0; i < hs.length; i++) { tr.appendChild(mk('th', null, hs[i])); }
    thead.appendChild(tr);
    tbl.appendChild(thead);
    var tb = mk('tbody');
    for (i = 0; i < list.length; i++) {
      var c = list[i];
      var r = mk('tr');
      r.appendChild(mk('td', 'k', c.name ? String(c.name) : '（未命名检查项）'));
      r.appendChild(mk('td', 'num', fmtFecVal(c.value)));
      r.appendChild(mk('td', 'num', fmtFecVal(c.target)));
      var ok = (c.pass === true);
      var bad = (c.pass === false);
      var mark = mk('td', ok ? 'check-ok' : (bad ? 'check-bad' : 'check-na'),
        ok ? '✓ 通过' : (bad ? '✗ 未通过' : '— 未知'));
      r.appendChild(mark);
      tb.appendChild(r);
    }
    tbl.appendChild(tb);
    host.appendChild(tbl);
  }

  /* ---- notes ---- */
  function renderFecNotes() {
    var host = el.fecNotes;
    if (!host) { return; }
    clear(host);
    var p = fecPayload();
    var arr = (p && Array.isArray(p.notes)) ? p.notes : [];
    setText('fecNoteCount', p ? (String(arr.length) + ' 条') : '—');
    if (!arr.length) {
      host.appendChild(mk('div', 'empty', p ? '报告未提供 notes' : fecEmptyText()));
      return;
    }
    var ul = mk('ul', 'warnlist');
    for (var i = 0; i < arr.length; i++) {
      var li = mk('li', 'bits-note');
      li.appendChild(mk('span', null, String(arr[i])));
      ul.appendChild(li);
    }
    host.appendChild(ul);
  }

  /* ---- 图例（点击开关曲线） ---- */
  function renderFecLegend() {
    var host = el.fecLegend;
    if (!host) { return; }
    clear(host);
    var p = fecPayload();
    var ids = fecCurveIds(p);
    var names = fecNameMap(p);
    for (var i = 0; i < ids.length; i++) {
      (function (id, color, name) {
        var off = !!state.ui.fecHidden[id];
        var btn = mk('button', 'legend-item is-toggle' + (off ? ' is-off' : ''));
        btn.type = 'button';
        btn.setAttribute('aria-pressed', off ? 'false' : 'true');
        btn.title = '点击' + (off ? '显示' : '隐藏') + '「' + name + '」曲线';
        var sw = mk('span', 'legend-swatch');
        sw.style.background = color;
        btn.appendChild(sw);
        btn.appendChild(mk('span', null, name));
        btn.addEventListener('click', function () {
          if (state.ui.fecHidden[id]) { delete state.ui.fecHidden[id]; }
          else { state.ui.fecHidden[id] = true; }
          fecHoverIndex = -1;
          if (el.fecTip) { show(el.fecTip, false); }
          renderFecLegend();
          drawFecCurves();
        });
        host.appendChild(btn);
      }(ids[i], fecColor(ids[i], i), names[ids[i]] || String(ids[i])));
    }
  }

  function renderFec() {
    renderFecCards();
    renderFecCatalogue();
    renderFecChecks();
    renderFecNotes();
    renderFecLegend();
  }

  /* ---- BER vs Eb/N0 曲线（本页重点，对数纵轴） ---- */
  function drawFecCurves() {
    var cv = el.fecCanvas;
    var emp = el.fecEmpty;
    var tip = el.fecTip;
    var p = fecPayload();
    var metric = fecMetricKey();
    var label = fecMetricLabel();
    var ids = fecCurveIds(p);
    var ebn0 = (p && Array.isArray(p.ebn0_db)) ? p.ebn0_db : null;
    var cvObj = fecCurves(p);

    function emptyState(msg) {
      fecGeom = null;
      if (tip) { show(tip, false); }
      if (emp) { emp.textContent = msg; show(emp, true); }
      if (el.fecMeta) { el.fecMeta.textContent = '—'; }
      if (el.fecPlotTitle) { el.fecPlotTitle.textContent = label + ' vs Eb/N0'; }
    }

    if (!cv) { return; }

    if (!p) { emptyState(fecEmptyText()); return; }
    if (!ebn0 || !ebn0.length) { emptyState('响应中没有 ebn0_db（无法确定横轴）'); return; }
    if (!cvObj || !ids.length) { emptyState('响应中没有 curves（或 curves 为空）'); return; }

    var i;
    var j;
    var k;
    var fullXmin = Infinity;
    var fullXmax = -Infinity;
    for (j = 0; j < ebn0.length; j++) {
      if (!isNum(ebn0[j])) { continue; }
      if (ebn0[j] < fullXmin) { fullXmin = ebn0[j]; }
      if (ebn0[j] > fullXmax) { fullXmax = ebn0[j]; }
    }

    var names = fecNameMap(p);
    var series = [];
    var yminData = Infinity;
    var ymaxData = 0;
    var totalPts = 0;
    var skipped = 0;
    var uneven = 0;
    var missing = 0;
    var hiddenN = 0;
    for (i = 0; i < ids.length; i++) {
      var id = ids[i];
      var cc = cvObj[id] || {};
      var arr = Array.isArray(cc[metric]) ? cc[metric] : null;
      if (!arr) { missing++; }
      else if (arr.length !== ebn0.length) { uneven++; }
      var lim = arr ? Math.min(arr.length, ebn0.length) : 0;
      var pts = [];
      for (j = 0; j < lim; j++) {
        var xv = ebn0[j];
        var yv = arr[j];
        if (!isNum(xv)) { continue; }
        if (!isNum(yv) || yv <= 0) { skipped++; continue; }
        pts.push({ i: j, x: xv, y: yv });
        if (yv < yminData) { yminData = yv; }
        if (yv > ymaxData) { ymaxData = yv; }
      }
      totalPts += pts.length;
      var hid = !!state.ui.fecHidden[id];
      if (hid) { hiddenN++; }
      series.push({ id: id, name: names[id] || String(id), color: fecColor(id, i), pts: pts, arr: arr, hidden: hid });
    }

    if (el.fecPlotTitle) { el.fecPlotTitle.textContent = label + ' vs Eb/N0'; }
    if (!totalPts) {
      emptyState('所选指标 ' + label + ' 没有可绘制的点（曲线缺失该指标或全部为 null）');
      return;
    }
    if (emp) { show(emp, false); }

    /* 纵轴：自适应向下取整到十进制档，最低 1e-6；最高不超过 1 */
    var ymin = Math.pow(10, Math.floor(Math.log10(yminData)));
    if (ymin < FEC_YMIN_FLOOR) { ymin = FEC_YMIN_FLOOR; }
    var ymax = Math.pow(10, Math.ceil(Math.log10(ymaxData)));
    if (ymax > 1) { ymax = 1; }
    if (!(ymax > ymin)) {
      if (ymin > FEC_YMIN_FLOOR) { ymin = Math.max(FEC_YMIN_FLOOR, ymin / 10); }
      if (!(ymax > ymin)) { ymax = Math.min(1, ymin * 10); }
    }

    /* 横轴：优先用完整 ebn0_db 的范围，保证只有个别曲线为 null 时坐标不跳 */
    var xmin = isNum(fullXmin) ? fullXmin : 0;
    var xmax = isNum(fullXmax) ? fullXmax : 1;
    if (!(xmax > xmin)) { xmin -= 0.5; xmax += 0.5; }
    else { var padx = (xmax - xmin) * 0.04; xmin -= padx; xmax += padx; }

    /* 交互层：Y 是对数刻度，窗口按 log10 缩放（不会被线性拉伸） */
    var win = chartWindow('fec', { x0: xmin, x1: xmax, y0: ymin, y1: ymax }, {
      yLog: true, xCount: ebn0.length, yCount: 0,
      fmtX: function (v) { return v.toFixed(2); }, unitX: 'dB',
      fmtY: fmtFecVal, unitY: '',
      canvas: cv, redraw: drawFecCurves
    });
    xmin = win.x0; xmax = win.x1; ymin = win.y0; ymax = win.y1;

    var f = fitCanvas(cv);
    var ctx = f.ctx;
    var m = { l: 66, r: 16, t: 12, b: 26 };
    var pw = f.w - m.l - m.r;
    var ph = f.h - m.t - m.b;
    if (pw < 32 || ph < 40) { fecGeom = null; return; }
    chartGeom('fec', m, pw, ph);

    var logY0 = Math.log10(ymin);
    var logY1 = Math.log10(ymax);
    function xOf(v) { return m.l + (v - xmin) / (xmax - xmin) * pw; }
    function yOf(v) {
      var t = (Math.log10(v) - logY0) / (logY1 - logY0);
      if (t < 0) { t = 0; }
      if (t > 1) { t = 1; }
      return m.t + ph - t * ph;
    }
    fecGeom = {
      xOf: xOf, yOf: yOf, xmin: xmin, xmax: xmax, m: m, pw: pw, ph: ph,
      ebn0: ebn0, series: series, metric: metric
    };

    /* 网格与刻度 */
    ctx.font = MONO_FONT_SM;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (var e = Math.round(logY0); e <= Math.round(logY1); e++) {
      var gv = Math.pow(10, e);
      if (gv < ymin - 1e-12 || gv > ymax + 1e-12) { continue; }
      var gy = Math.round(yOf(gv)) + 0.5;
      ctx.strokeStyle = (Math.abs(gv - 1) < 1e-12) ? COL.gridStrong : COL.grid;
      ctx.beginPath();
      ctx.moveTo(m.l, gy);
      ctx.lineTo(m.l + pw, gy);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText('1e' + String(e), m.l - 6, gy);
    }
    var xt = ticks(xmin, xmax, 6);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (i = 0; i < xt.length; i++) {
      var gx = Math.round(xOf(xt[i])) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(gx, m.t);
      ctx.lineTo(gx, m.t + ph);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(xt[i].toFixed(1), gx, m.t + ph + 5);
    }

    /* 悬停十字线 */
    var hoverX = null;
    if (fecHoverIndex >= 0 && fecHoverIndex < ebn0.length && isNum(ebn0[fecHoverIndex])) {
      hoverX = xOf(ebn0[fecHoverIndex]);
      if (hoverX < m.l - 1 || hoverX > m.l + pw + 1) { hoverX = null; }
    }
    if (hoverX !== null) {
      ctx.save();
      ctx.strokeStyle = COL.gridStrong;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(Math.round(hoverX) + 0.5, m.t);
      ctx.lineTo(Math.round(hoverX) + 0.5, m.t + ph);
      ctx.stroke();
      ctx.restore();
    }

    /* 曲线：逐点容错，null/<=0 断开但不算断线错误 */
    var visible = 0;
    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, pw, ph);
    ctx.clip();
    for (i = 0; i < series.length; i++) {
      var s = series[i];
      if (s.hidden || !s.pts.length) { continue; }
      visible++;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.4;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      var started = false;
      for (k = 0; k < s.pts.length; k++) {
        var px = xOf(s.pts[k].x);
        var py = yOf(s.pts[k].y);
        if (!started) { ctx.moveTo(px, py); started = true; } else { ctx.lineTo(px, py); }
      }
      ctx.stroke();
      ctx.lineWidth = 1;
      ctx.fillStyle = s.color;
      for (k = 0; k < s.pts.length; k++) {
        ctx.beginPath();
        ctx.arc(xOf(s.pts[k].x), yOf(s.pts[k].y), 1.9, 0, Math.PI * 2);
        ctx.fill();
      }
      if (hoverX !== null) {
        for (k = 0; k < s.pts.length; k++) {
          if (s.pts[k].i !== fecHoverIndex) { continue; }
          ctx.beginPath();
          ctx.arc(xOf(s.pts[k].x), yOf(s.pts[k].y), 3.2, 0, Math.PI * 2);
          ctx.strokeStyle = COL.peak;
          ctx.lineWidth = 1.2;
          ctx.stroke();
          ctx.lineWidth = 1;
        }
      }
    }
    ctx.restore();

    ctx.fillStyle = COL.axisText2;
    ctx.font = MONO_FONT_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(label + '（对数）', m.l + 2, m.t + 2);
    ctx.textAlign = 'right';
    ctx.fillText('Eb/N0 (dB)', m.l + pw, m.t + ph + 5);
    strokeFrame(ctx, m.l, m.t, pw, ph);
    drawViewBadge(ctx, 'fec', m, pw, ph);

    if (el.fecMeta) {
      var bits = [];
      if (p.updated_at) { bits.push('更新 ' + String(p.updated_at)); }
      if (p.report_path) { bits.push(String(p.report_path)); }
      bits.push(label + ' · ' + String(visible) + '/' + String(series.length) + ' 条曲线可见');
      bits.push('Eb/N0 ' + fmtInt(ebn0.length) + ' 点');
      bits.push('有效 ' + fmtInt(totalPts) + ' 点');
      if (skipped) { bits.push('未绘 ' + fmtInt(skipped) + ' 点（null 或 ≤0）'); }
      if (uneven) { bits.push(String(uneven) + ' 条曲线长度不等'); }
      if (missing) { bits.push(String(missing) + ' 条曲线缺 ' + label); }
      el.fecMeta.textContent = bits.join('  ·  ');
    }
  }

  /* ---- 悬停 tooltip ---- */
  function fecTipShow(clientX, clientY) {
    var tip = el.fecTip;
    var body = el.fecPlotBody;
    var g = fecGeom;
    if (!tip || !body || !g) { return; }
    var idx = fecHoverIndex;
    if (idx < 0 || !isNum(g.ebn0[idx])) { show(tip, false); return; }
    clear(tip);
    tip.appendChild(mk('span', 'tip-k', 'Eb/N0 ' + Number(g.ebn0[idx]).toFixed(2) + ' dB'));
    var shown = 0;
    for (var i = 0; i < g.series.length; i++) {
      var s = g.series[i];
      if (s.hidden) { continue; }
      var row = mk('div', 'tip-row');
      var left = mk('span');
      var sw = mk('span', 'tip-swatch');
      sw.style.background = s.color;
      left.appendChild(sw);
      left.appendChild(mk('span', null, s.name));
      row.appendChild(left);
      row.appendChild(mk('span', null, (s.arr && isNum(s.arr[idx])) ? fmtFecBer(s.arr[idx]) : '—'));
      tip.appendChild(row);
      shown++;
    }
    if (!shown) { show(tip, false); return; }
    show(tip, true);
    var rect = body.getBoundingClientRect();
    var w = tip.offsetWidth || 200;
    var x = clientX - rect.left + 14;
    if (x + w > rect.width - 4) { x = Math.max(4, clientX - rect.left - w - 14); }
    var y = Math.max(4, clientY - rect.top - 8);
    tip.style.left = Math.round(x) + 'px';
    tip.style.top = Math.round(y) + 'px';
  }

  function fecOnMove(ev) {
    var g = fecGeom;
    var cv = el.fecCanvas;
    if (!g || !cv || !g.ebn0) { return; }
    /* 拖动平移时不要弹 tooltip，避免和缩放/平移打架 */
    if (chartDragging()) { return; }
    var rect = cv.getBoundingClientRect();
    var x = ev.clientX - rect.left;
    var best = -1;
    var bd = Infinity;
    for (var i = 0; i < g.ebn0.length; i++) {
      if (!isNum(g.ebn0[i])) { continue; }
      var d = Math.abs(g.xOf(g.ebn0[i]) - x);
      if (d < bd) { bd = d; best = i; }
    }
    if (bd > 24) { best = -1; }
    if (best !== fecHoverIndex) {
      fecHoverIndex = best;
      drawFecCurves();
    }
    if (fecHoverIndex >= 0) { fecTipShow(ev.clientX, ev.clientY); }
    else if (el.fecTip) { show(el.fecTip, false); }
  }

  function fecOnLeave() {
    fecHoverIndex = -1;
    if (el.fecTip) { show(el.fecTip, false); }
    drawFecCurves();
  }

  /* ================= 谱相关（SSCA，M8） =================
   * 契约 §13：GET /api/ssca?path=&nfft=&max_points=
   * surface_db 是 A 行 × F 列；freqs 长度 = 每行长度 = freq_profile 长度；
   * alphas 长度 = 行数 = alpha_profile 长度。null 表示无效点，必须逐点容错。
   */

  var sscaOff = null;
  var sscaMapCache = null;
  var sscaHover = null;

  function sscaData() { return state.ssca; }

  function sscaEmptyText(what) {
    if (state.failed.ssca) { return '谱相关加载失败（见顶部错误条）'; }
    if (state.pending.ssca) { return '加载中…'; }
    if (!state.path) { return '等待数据：请在左侧选择案例或输入路径'; }
    return what || '暂无谱相关数据';
  }

  function loadSsca(mockFile) {
    if (!state.path) { return; }
    var gen = state.dataGen;
    state.pending.ssca = true;
    state.failed.ssca = false;
    renderAll();
    var params = { path: state.path, nfft: state.ui.sscaNfft, max_points: state.ui.sscaMaxPoints };
    apiGet('ssca', params, mockFile).then(function (d) {
      if (gen !== state.dataGen) { return; }
      state.pending.ssca = false;
      state.failed.ssca = false;
      state.updatedAt = nowStamp();
      state.ssca = d;
      renderAll();
    }, function (err) {
      if (gen !== state.dataGen) { return; }
      /* mock 兜底：<slug>.ssca.json 不存在（404）时改读 /static/mock/ssca.json */
      if (state.mock && !mockFile && err.status === 404) { loadSsca('ssca'); return; }
      state.pending.ssca = false;
      state.failed.ssca = true;
      recordError('ssca', err.status, err.message, err.url);
      renderAll();
    });
  }

  function getSscaOffscreen(w, h) {
    if (!sscaOff) { sscaOff = document.createElement('canvas'); }
    if (sscaOff.width !== w || sscaOff.height !== h) { sscaOff.width = w; sscaOff.height = h; }
    return sscaOff;
  }

  /* 解析热图实际使用的两条轴。
     契约 §13 要求 freqs 长度 == 每行 surface_db 长度；实测后端额外给出 surface_freqs /
     surface_alphas，且 freqs(=freq_profile 轴, 256) 比 surface 列数(=128) 长。
     因此：surface_* 与 surface_db 形状一致时优先用 surface_*，否则退回 freqs/alphas 并截断，
     保证"轴长度 = 数据维度"，既不崩也不把图拉歪。 */
  function sscaSurfaceAxes(d) {
    var rows = (d && Array.isArray(d.surface_db)) ? d.surface_db : [];
    var cols = 0;
    var r;
    for (r = 0; r < rows.length; r++) {
      if (Array.isArray(rows[r]) && rows[r].length > cols) { cols = rows[r].length; }
    }
    var sf = (d && Array.isArray(d.surface_freqs)) ? d.surface_freqs : null;
    var sa = (d && Array.isArray(d.surface_alphas)) ? d.surface_alphas : null;
    var fx = (sf && sf.length === cols && cols > 1) ? sf : ((d && Array.isArray(d.freqs)) ? d.freqs : []);
    var ay = (sa && sa.length === rows.length && rows.length > 1) ? sa : ((d && Array.isArray(d.alphas)) ? d.alphas : []);
    if (fx.length > cols) { fx = fx.slice(0, cols); }
    if (ay.length > rows.length) { ay = ay.slice(0, rows.length); }
    return { freqs: fx, alphas: ay, cols: cols, rows: rows.length };
  }

  function sscaMapKey(d, F, A, fx0, fx1, a0, a1) {
    return [d.path, d.nfft, F, A, d.min_db, d.max_db, fx0, fx1, a0, a1].join('|');
  }

  /* 双频平面位图（独立离屏画布，避免和瀑布图共用同一块 offCanvas 互相清空） */
  function sscaBuildImage(d, F, A, minDb, maxDb, fx0, fx1, a0, a1) {
    var key = sscaMapKey(d, F, A, fx0, fx1, a0, a1);
    if (sscaMapCache && sscaMapCache.key === key && sscaMapCache.off
      && sscaMapCache.off.width === F && sscaMapCache.off.height === A) {
      return sscaMapCache.off;
    }
    var off = getSscaOffscreen(F, A);
    var octx = off.getContext('2d');
    var img = octx.createImageData(F, A);
    var span = (maxDb - minDb) || 1;
    var r;
    var c;
    for (r = 0; r < A; r++) {
      var row = Array.isArray(d.surface_db[r]) ? d.surface_db[r] : null;
      var iy = A - 1 - r; /* α 越大越靠上，与 yOf 的方向一致 */
      for (c = 0; c < F; c++) {
        var v = row ? row[c] : null;
        var t = isNum(v) ? (v - minDb) / span : 0;
        if (t < 0) { t = 0; }
        if (t > 1) { t = 1; }
        var rgb = cmap(t);
        var p = (iy * F + c) * 4;
        img.data[p] = rgb[0];
        img.data[p + 1] = rgb[1];
        img.data[p + 2] = rgb[2];
        img.data[p + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    sscaMapCache = { key: key, off: off };
    return off;
  }

  function drawSscaMap() {
    var cv = el.sscaCanvas;
    if (!cv) { return; }
    var emp = el.sscaEmpty;
    var d = sscaData();
    var ax = d ? sscaSurfaceAxes(d) : { freqs: [], alphas: [], cols: 0, rows: 0 };
    var freqs = ax.freqs;
    var alphas = ax.alphas;
    var F = freqs.length;
    var A = alphas.length;
    var ok = !!(d && Array.isArray(d.surface_db) && d.surface_db.length > 0 && F > 1 && A > 1);
    if (!ok) {
      if (emp) { emp.textContent = sscaEmptyText(); show(emp, true); }
      if (el.sscaMapMeta) { el.sscaMapMeta.textContent = '—'; }
      if (el.sscaTip) { show(el.sscaTip, false); }
      return;
    }
    if (emp) { show(emp, false); }

    var minDb = isNum(d.min_db) ? d.min_db : -60;
    var maxDb = isNum(d.max_db) ? d.max_db : 0;
    if (!(maxDb > minDb)) { maxDb = minDb + 1; }

    var f = fitCanvas(cv);
    var ctx = f.ctx;
    var m = { l: 62, r: 66, t: 12, b: 24 };
    var pw = f.w - m.l - m.r;
    var ph = f.h - m.t - m.b;
    if (pw < 32 || ph < 32) { return; }

    var fxAll0 = freqs[0];
    var fxAll1 = freqs[F - 1];
    if (!isNum(fxAll0) || !isNum(fxAll1) || !(fxAll1 > fxAll0)) { fxAll0 = 0; fxAll1 = 1; }
    var aAll0 = isNum(alphas[0]) ? alphas[0] : 0;
    var aAll1 = isNum(alphas[A - 1]) ? alphas[A - 1] : 1;
    if (!(aAll1 > aAll0)) { aAll1 = aAll0 + 1; }

    /* 交互层：X = 频移 f，Y = 循环频率 α（二维同时缩放） */
    var win = chartWindow('sscaMap', { x0: fxAll0, x1: fxAll1, y0: aAll0, y1: aAll1 }, {
      xCount: F, yCount: A,
      fmtX: fmtHzTick, unitX: 'Hz', fmtY: fmtHzTick, unitY: 'Hz',
      canvas: cv, redraw: drawSscaMap
    });
    chartGeom('sscaMap', m, pw, ph);

    var off = sscaBuildImage(d, F, A, minDb, maxDb, fxAll0, fxAll1, aAll0, aAll1);
    ctx.imageSmoothingEnabled = false;
    var colA = (win.x0 - fxAll0) / (fxAll1 - fxAll0) * (F - 1);
    var colB = (win.x1 - fxAll0) / (fxAll1 - fxAll0) * (F - 1);
    var rowLo = (win.y0 - aAll0) / (aAll1 - aAll0) * (A - 1);
    var rowHi = (win.y1 - aAll0) / (aAll1 - aAll0) * (A - 1);
    var sw = (colB - colA) + 1;
    var sh = (rowHi - rowLo) + 1;
    var sy = (A - 1) - rowHi;
    if (colA < 0) { colA = 0; }
    if (sy < 0) { sy = 0; }
    if (sw < 1) { sw = 1; }
    if (sh < 1) { sh = 1; }
    if (colA + sw > F) { sw = F - colA; }
    if (sy + sh > A) { sh = A - sy; }
    if (sw > 0 && sh > 0) { ctx.drawImage(off, colA, sy, sw, sh, m.l, m.t, pw, ph); }

    function xOf(v) { return m.l + (v - win.x0) / (win.x1 - win.x0) * pw; }
    function yOf(v) { return m.t + ph - (v - win.y0) / (win.y1 - win.y0) * ph; }

    /* 网格与刻度（跟随窗口） */
    ctx.font = MONO_FONT_SM;
    ctx.lineWidth = 1;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var xt = ticks(win.x0, win.x1, 5);
    var i;
    for (i = 0; i < xt.length; i++) {
      var xx = Math.round(xOf(xt[i])) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(xx, m.t);
      ctx.lineTo(xx, m.t + ph);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(fmtHzTick(xt[i]), xx, m.t + ph + 5);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    var yt = ticks(win.y0, win.y1, 5);
    for (i = 0; i < yt.length; i++) {
      var yy = Math.round(yOf(yt[i])) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(m.l, yy);
      ctx.lineTo(m.l + pw, yy);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(fmtHzTick(yt[i]), m.l - 6, yy);
    }

    /* 峰值：十字 + 圆圈 + α 标注 */
    var peaks = Array.isArray(d.peaks) ? d.peaks : [];
    var marked = 0;
    for (i = 0; i < peaks.length; i++) {
      var pk = (peaks[i] && typeof peaks[i] === 'object') ? peaks[i] : {};
      if (!isNum(pk.freq_hz) || !isNum(pk.alpha_hz)) { continue; }
      var pxp = xOf(pk.freq_hz);
      var pyp = yOf(pk.alpha_hz);
      if (pxp < m.l || pxp > m.l + pw || pyp < m.t || pyp > m.t + ph) { continue; }
      marked++;
      ctx.save();
      ctx.strokeStyle = COL.peak;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pxp - 6, pyp);
      ctx.lineTo(pxp + 6, pyp);
      ctx.moveTo(pxp, pyp - 6);
      ctx.lineTo(pxp, pyp + 6);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(pxp, pyp, 4, 0, Math.PI * 2);
      ctx.stroke();
      var lbl = 'α ' + fmtHzTick(pk.alpha_hz) + 'Hz';
      if (isNum(pk.level_db)) { lbl += ' ' + pk.level_db.toFixed(1) + 'dB'; }
      var tw = ctx.measureText(lbl).width;
      var tx = pxp + 9;
      if (tx + tw > m.l + pw) { tx = pxp - 9 - tw; }
      if (tx < m.l) { tx = m.l + 2; }
      var ty = pyp - 6;
      if (ty < m.t + 12) { ty = pyp + 12; }
      ctx.fillStyle = 'rgba(10, 14, 19, 0.78)';
      ctx.fillRect(tx - 2, ty - 11, tw + 4, 12);
      ctx.fillStyle = COL.peak;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(lbl, tx, ty);
      ctx.restore();
    }

    /* 鼠标悬停十字（tooltip 由 DOM 负责） */
    if (sscaHover && isNum(sscaHover.freq_hz) && isNum(sscaHover.alpha_hz)) {
      var hx = xOf(sscaHover.freq_hz);
      var hy = yOf(sscaHover.alpha_hz);
      if (hx >= m.l && hx <= m.l + pw && hy >= m.t && hy <= m.t + ph) {
        ctx.save();
        ctx.strokeStyle = COL.peak;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(Math.round(hx) + 0.5, m.t);
        ctx.lineTo(Math.round(hx) + 0.5, m.t + ph);
        ctx.moveTo(m.l, Math.round(hy) + 0.5);
        ctx.lineTo(m.l + pw, Math.round(hy) + 0.5);
        ctx.stroke();
        ctx.restore();
      }
    }

    /* 色标（min_db…max_db） */
    var cbX = m.l + pw + 12;
    var cbW = 11;
    var g = ctx.createLinearGradient(0, m.t + ph, 0, m.t);
    g.addColorStop(0, rgbCss(cmap(0)));
    g.addColorStop(1, rgbCss(cmap(1)));
    ctx.fillStyle = g;
    ctx.fillRect(cbX, m.t, cbW, ph);
    ctx.strokeStyle = COL.frame;
    ctx.strokeRect(cbX + 0.5, m.t + 0.5, cbW - 1, ph - 1);
    ctx.font = MONO_FONT_SM;
    ctx.fillStyle = COL.axisText;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(maxDb.toFixed(0), cbX + cbW + 4, m.t + 5);
    ctx.fillText(((maxDb + minDb) / 2).toFixed(0), cbX + cbW + 4, m.t + ph / 2);
    ctx.fillText(minDb.toFixed(0), cbX + cbW + 4, m.t + ph - 5);
    ctx.textBaseline = 'top';
    ctx.fillText('dB', cbX + cbW + 4, m.t + 14);

    ctx.fillStyle = COL.axisText2;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('循环频率 α (Hz)', m.l + 2, m.t + 2);
    ctx.textAlign = 'right';
    ctx.fillText('频移 f (Hz)', m.l + pw, m.t + ph + 5);

    strokeFrame(ctx, m.l, m.t, pw, ph);
    drawViewBadge(ctx, 'sscaMap', m, pw, ph);

    if (el.sscaMapMeta) {
      var parts = ['热图 ' + fmtInt(A) + ' α × ' + fmtInt(F) + ' f'];
      parts.push('色标 ' + minDb.toFixed(1) + '…' + maxDb.toFixed(1) + ' dB');
      parts.push('峰值标注 ' + fmtInt(marked) + '/' + fmtInt(peaks.length));
      if (Array.isArray(d.freqs) && d.freqs.length !== F) { parts.push('freqs ' + fmtInt(d.freqs.length) + '（f 剖面轴）'); }
      if (isNum(d.symbol_rate)) { parts.push('symbol_rate ' + fmtCompact(d.symbol_rate) + 'Hz'); }
      el.sscaMapMeta.textContent = parts.join('  ·  ');
    }
  }

  /* α 剖面 / f 剖面：同一条折线绘制逻辑，各自独立的缩放视图 */
  function drawSscaProfile(cfg) {
    var cv = cfg.canvas;
    if (!cv) { return; }
    var d = sscaData();
    var prof = (d && cfg.prof && typeof cfg.prof === 'object') ? cfg.prof : null;
    var xs = (prof && Array.isArray(prof[cfg.xKey])) ? prof[cfg.xKey] : [];
    var ys = (prof && Array.isArray(prof[cfg.yKey])) ? prof[cfg.yKey] : [];
    var lim = Math.min(xs.length, ys.length);
    var pts = [];
    var i;
    for (i = 0; i < lim; i++) {
      if (isNum(xs[i]) && isNum(ys[i])) { pts.push({ x: xs[i], y: ys[i] }); }
    }
    if (!pts.length) {
      if (cfg.empty) { cfg.empty.textContent = sscaEmptyText(cfg.label + '剖面无有效点'); show(cfg.empty, true); }
      if (cfg.meta) { cfg.meta.textContent = '—'; }
      return;
    }
    if (cfg.empty) { show(cfg.empty, false); }

    var x0 = pts[0].x;
    var x1 = pts[0].x;
    var ylo = pts[0].y;
    var yhi = pts[0].y;
    for (i = 0; i < pts.length; i++) {
      if (pts[i].x < x0) { x0 = pts[i].x; }
      if (pts[i].x > x1) { x1 = pts[i].x; }
      if (pts[i].y < ylo) { ylo = pts[i].y; }
      if (pts[i].y > yhi) { yhi = pts[i].y; }
    }
    if (!(x1 > x0)) { x1 = x0 + 1; }
    var padY = (yhi - ylo) * 0.12;
    if (!(padY > 0)) { padY = 1; }
    ylo -= padY;
    yhi += padY;

    var f = fitCanvas(cv);
    var ctx = f.ctx;
    var m = { l: 58, r: 16, t: 12, b: 24 };
    var pw = f.w - m.l - m.r;
    var ph = f.h - m.t - m.b;
    if (pw < 24 || ph < 24) { return; }

    var win = chartWindow(cfg.id, { x0: x0, x1: x1, y0: ylo, y1: yhi }, {
      xCount: pts.length, yCount: 0,
      fmtX: fmtHzTick, unitX: 'Hz',
      fmtY: function (v) { return v.toFixed(0); }, unitY: 'dB',
      canvas: cv, redraw: cfg.redraw
    });
    chartGeom(cfg.id, m, pw, ph);

    function xOf(v) { return m.l + (v - win.x0) / (win.x1 - win.x0) * pw; }
    function yOf(v) { return m.t + ph - (v - win.y0) / (win.y1 - win.y0) * ph; }

    ctx.font = MONO_FONT_SM;
    ctx.lineWidth = 1;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var xt = ticks(win.x0, win.x1, 5);
    for (i = 0; i < xt.length; i++) {
      var gx = Math.round(xOf(xt[i])) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(gx, m.t);
      ctx.lineTo(gx, m.t + ph);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(fmtHzTick(xt[i]), gx, m.t + ph + 5);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    var yt = ticks(win.y0, win.y1, 4);
    for (i = 0; i < yt.length; i++) {
      var gy = Math.round(yOf(yt[i])) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath();
      ctx.moveTo(m.l, gy);
      ctx.lineTo(m.l + pw, gy);
      ctx.stroke();
      ctx.fillStyle = COL.axisText;
      ctx.fillText(yt[i].toFixed(0), m.l - 6, gy);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, pw, ph);
    ctx.clip();
    ctx.strokeStyle = cfg.color;
    ctx.lineWidth = 1.3;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (i = 0; i < pts.length; i++) {
      var lx = xOf(pts[i].x);
      var ly = yOf(pts[i].y);
      if (i === 0) { ctx.moveTo(lx, ly); } else { ctx.lineTo(lx, ly); }
    }
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.restore();

    ctx.fillStyle = COL.axisText2;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(cfg.ylabel, m.l + 2, m.t + 2);
    ctx.textAlign = 'right';
    ctx.fillText(cfg.xlabel, m.l + pw, m.t + ph + 5);
    strokeFrame(ctx, m.l, m.t, pw, ph);
    drawViewBadge(ctx, cfg.id, m, pw, ph);

    if (cfg.meta) {
      cfg.meta.textContent = cfg.label + ' ' + fmtInt(pts.length) + ' 点  ·  '
        + fmtHzTick(x0) + '…' + fmtHzTick(x1) + ' Hz';
    }
  }

  function drawSscaAlpha() {
    var d = sscaData();
    drawSscaProfile({
      id: 'sscaAlpha', canvas: el.sscaAlphaCanvas, empty: el.sscaAlphaEmpty, meta: el.sscaAlphaMeta,
      prof: d ? d.alpha_profile : null, xKey: 'alpha_hz', yKey: 'level_db',
      label: 'α 剖面（对 f 取最大）', xlabel: '循环频率 α (Hz)', ylabel: '电平 (dB)',
      color: COL.line, redraw: drawSscaAlpha
    });
  }

  function drawSscaFreq() {
    var d = sscaData();
    drawSscaProfile({
      id: 'sscaFreq', canvas: el.sscaFreqCanvas, empty: el.sscaFreqEmpty, meta: el.sscaFreqMeta,
      prof: d ? d.freq_profile : null, xKey: 'freq_hz', yKey: 'level_db',
      label: 'f 剖面（对 α 取最大）', xlabel: '频移 f (Hz)', ylabel: '电平 (dB)',
      color: COL.syncFreq, redraw: drawSscaFreq
    });
  }

  function sscaHoverClear() {
    var had = !!sscaHover;
    sscaHover = null;
    if (had) { drawSscaMap(); }
    if (el.sscaTip) { show(el.sscaTip, false); }
  }

  function sscaTipShow(clientX, clientY) {
    var tip = el.sscaTip;
    var body = (el.sscaCanvas && el.sscaCanvas.parentNode) ? el.sscaCanvas.parentNode : null;
    if (!tip || !body || !sscaHover) { return; }
    clear(tip);
    tip.appendChild(mk('span', 'tip-k', 'α ' + fmtCompact(sscaHover.alpha_hz) + ' Hz  ·  f ' + fmtCompact(sscaHover.freq_hz) + ' Hz'));
    var row = mk('div', 'tip-row');
    row.appendChild(mk('span', null, 'SSCA 电平'));
    row.appendChild(mk('span', null, isNum(sscaHover.level_db) ? (sscaHover.level_db.toFixed(2) + ' dB') : 'null（无效点）'));
    tip.appendChild(row);
    show(tip, true);
    var rect = body.getBoundingClientRect();
    var w = tip.offsetWidth || 200;
    var x = clientX - rect.left + 14;
    if (x + w > rect.width - 4) { x = Math.max(4, clientX - rect.left - w - 14); }
    var y = Math.max(4, clientY - rect.top - 8);
    tip.style.left = Math.round(x) + 'px';
    tip.style.top = Math.round(y) + 'px';
  }

  function sscaOnMove(ev) {
    var cv = el.sscaCanvas;
    var v = viewOf('sscaMap');
    if (!cv || !v.m || !v.win || dragV) { return; }
    var d = sscaData();
    if (!d) { return; }
    var rect = cv.getBoundingClientRect();
    var mx = ev.clientX - rect.left;
    var my = ev.clientY - rect.top;
    if (mx < v.m.l || mx > v.m.l + v.pw || my < v.m.t || my > v.m.t + v.ph) { sscaHoverClear(); return; }
    var fv = v.win.x0 + (mx - v.m.l) / v.pw * (v.win.x1 - v.win.x0);
    var av = v.win.y0 + (v.m.t + v.ph - my) / v.ph * (v.win.y1 - v.win.y0);
    /* 必须用热图实际使用的 surface 轴，否则真机上 freqs(256) 与 surface 列数(128) 不一致，
       取到的列号会越界，tooltip 会把有效点错报成 null */
    var ax = sscaSurfaceAxes(d);
    var freqs = ax.freqs;
    var alphas = ax.alphas;
    var c = nearestIndex(freqs, fv);
    var r = nearestIndex(alphas, av);
    if (c < 0 || r < 0) { sscaHoverClear(); return; }
    var row = Array.isArray(d.surface_db[r]) ? d.surface_db[r] : null;
    var lvl = row ? row[c] : null;
    sscaHover = { freq_hz: freqs[c], alpha_hz: alphas[r], level_db: isNum(lvl) ? lvl : null };
    drawSscaMap();
    sscaTipShow(ev.clientX, ev.clientY);
  }

  function renderSscaTop() {
    if (!el.sscaMeta) { return; }
    var d = sscaData();
    if (!d) { el.sscaMeta.textContent = sscaEmptyText(); return; }
    var bits = [];
    bits.push('nfft ' + fmtInt(isNum(d.nfft) ? d.nfft : state.ui.sscaNfft));
    bits.push('max_points ' + fmtInt(state.ui.sscaMaxPoints));
    if (isNum(d.sample_rate)) { bits.push('Fs ' + fmtCompact(d.sample_rate) + 'Hz'); }
    if (isNum(d.symbol_rate)) { bits.push('symbol_rate ' + fmtCompact(d.symbol_rate) + 'Hz'); }
    var ax = sscaSurfaceAxes(d);
    bits.push('热图 ' + fmtInt(ax.rows) + ' α × ' + fmtInt(ax.cols) + ' f');
    el.sscaMeta.textContent = bits.join('  ·  ');
  }

  function renderSscaLegend() {
    var host = el.sscaLegend;
    if (!host) { return; }
    clear(host);
    var d = sscaData();
    var items = [
      ['legend-swatch is-band', 'SSCA 幅度色标'],
      ['legend-swatch is-peak', '峰值 (α/f)']
    ];
    if (d && isNum(d.min_db) && isNum(d.max_db)) {
      items.push(['legend-swatch is-ssca-scale', fmtFixed(d.min_db, 1) + ' … ' + fmtFixed(d.max_db, 1) + ' dB']);
      items[0][1] = '幅度 (dB)';
    }
    for (var i = 0; i < items.length; i++) {
      var n = mk('span', 'legend-item');
      n.appendChild(mk('span', items[i][0]));
      n.appendChild(mk('span', null, items[i][1]));
      host.appendChild(n);
    }
  }

  /* 谱相关 vs 参数估计：同一页已有 analyze 就直接对照 */
  function renderSscaFeatures() {
    var host = el.sscaFeatures;
    if (!host) { return; }
    clear(host);
    var d = sscaData();
    if (!d) { host.appendChild(mk('div', 'empty', sscaEmptyText())); return; }
    var ft = (d.features && typeof d.features === 'object') ? d.features : null;
    var a = state.analyze;
    var em = (d.estimate && typeof d.estimate === 'object') ? d.estimate : null;
    var estRs = (a && a.symbol_rate && isNum(a.symbol_rate.value)) ? a.symbol_rate.value : null;
    var estCfo = (a && a.cfo && isNum(a.cfo.value)) ? a.cfo.value : null;
    var estLabel = '参数估计';
    /* /api/analyze 还没回来时，退回 /api/ssca 自带的内嵌 estimate，并注明来源 */
    if (estRs === null && estCfo === null && em) {
      estRs = isNum(em.symbol_rate_hz) ? em.symbol_rate_hz : null;
      estCfo = isNum(em.cfo_hz) ? em.cfo_hz : null;
      if (estRs !== null || estCfo !== null) { estLabel = '内嵌 estimate'; }
    }
    var rows = [
      { k: '符号速率', sub: 'features.symbol_rate_hz', ssca: ft && isNum(ft.symbol_rate_hz) ? ft.symbol_rate_hz : null, est: estRs, unit: 'Hz' },
      { k: '载频偏移', sub: 'features.cfo_hz', ssca: ft && isNum(ft.cfo_hz) ? ft.cfo_hz : null, est: estCfo, unit: 'Hz' }
    ];
    var tbl = mk('table', 'fec-table');
    var thead = mk('thead');
    var tr = mk('tr');
    var hs = ['特征', '谱相关', '参数估计 (' + estLabel + ')', '差值', '相对差'];
    for (var i = 0; i < hs.length; i++) { tr.appendChild(mk('th', null, hs[i])); }
    thead.appendChild(tr);
    tbl.appendChild(thead);
    var tb = mk('tbody');
    for (i = 0; i < rows.length; i++) {
      var rw = rows[i];
      var r = mk('tr');
      var kTd = mk('td', 'k', rw.k);
      kTd.title = rw.sub;
      r.appendChild(kTd);
      r.appendChild(mk('td', 'rate', isNum(rw.ssca) ? (fmtEst(rw.ssca) + ' ' + rw.unit) : '—'));
      r.appendChild(mk('td', 'v-true', isNum(rw.est) ? (fmtEst(rw.est) + ' ' + rw.unit) : '—（analyze / estimate 均未返回）'));
      var diff = (isNum(rw.ssca) && isNum(rw.est)) ? (rw.ssca - rw.est) : null;
      r.appendChild(mk('td', 'num', isNum(diff) ? (signed(diff, 3) + ' ' + rw.unit) : '—'));
      var rel = (isNum(diff) && rw.est !== 0) ? (diff / rw.est * 100) : null;
      r.appendChild(mk('td', 'num', isNum(rel) ? (signed(rel, 3) + '%') : '—'));
      tb.appendChild(r);
    }
    tbl.appendChild(tb);
    host.appendChild(tbl);
    var note = (ft && ft.note) ? String(ft.note) : 'α 轴峰 = 符号速率及其谐波；f 轴峰 = 载频偏移';
    host.appendChild(mk('p', 'hint', '谱相关 vs 参数估计：' + note));
  }

  function renderSscaPeaks() {
    var host = el.sscaPeaks;
    if (!host) { return; }
    clear(host);
    var d = sscaData();
    if (!d) {
      setText('sscaPeakCount', '—');
      host.appendChild(mk('div', 'empty', sscaEmptyText()));
      return;
    }
    var peaks = Array.isArray(d.peaks) ? d.peaks : [];
    setText('sscaPeakCount', fmtInt(peaks.length) + ' 个');
    if (!peaks.length) { host.appendChild(mk('div', 'empty', '无显著局部峰（peaks 为空）')); return; }
    var tbl = mk('table', 'fec-table');
    var thead = mk('thead');
    var tr = mk('tr');
    var hs = ['#', 'α (Hz)', 'f (Hz)', '电平 (dB)'];
    for (var i = 0; i < hs.length; i++) { tr.appendChild(mk('th', null, hs[i])); }
    thead.appendChild(tr);
    tbl.appendChild(thead);
    var tb = mk('tbody');
    for (i = 0; i < peaks.length; i++) {
      var pk = (peaks[i] && typeof peaks[i] === 'object') ? peaks[i] : {};
      var r = mk('tr');
      r.appendChild(mk('td', 'num', String(i + 1)));
      r.appendChild(mk('td', 'k', isNum(pk.alpha_hz) ? fmtEst(pk.alpha_hz) : '—'));
      r.appendChild(mk('td', 'num', isNum(pk.freq_hz) ? fmtEst(pk.freq_hz) : '—'));
      r.appendChild(mk('td', 'num', isNum(pk.level_db) ? pk.level_db.toFixed(2) : '—'));
      tb.appendChild(r);
    }
    tbl.appendChild(tb);
    host.appendChild(tbl);
  }

  function renderSscaWarnings() {
    var host = el.sscaWarnings;
    if (!host) { return; }
    clear(host);
    var d = sscaData();
    if (!d) {
      setText('sscaWarnCount', '—');
      host.appendChild(mk('div', 'empty', sscaEmptyText()));
      return;
    }
    var list = Array.isArray(d.warnings) ? d.warnings : [];
    setText('sscaWarnCount', fmtInt(list.length) + ' 条');
    if (!list.length) { host.appendChild(mk('div', 'empty', '无告警')); return; }
    var ul = mk('ul', 'warnlist');
    for (var i = 0; i < list.length; i++) { ul.appendChild(mk('li', null, String(list[i]))); }
    host.appendChild(ul);
  }

  function renderSsca() {
    renderSscaTop();
    renderSscaLegend();
    renderSscaFeatures();
    renderSscaPeaks();
    renderSscaWarnings();
  }

  /* ================= 报告（Narrate，M9 /api/narrate） =================
   * 契约 §16：GET /api/narrate?path=&provider=offline|auto&max_tokens=800
   * mode=offline 是确定性离线渲染，llm 是真实模型输出；两者输出结构完全一致。
   * sections[].evidence 是"LLM 不许编"的落点：逐行 key/value/unit/source/confidence。
   */

  function narrateData() { return state.narrate; }

  function narrateEmptyText() {
    if (state.failed.narrate) { return '报告生成失败（见顶部错误条）'; }
    if (state.pending.narrate) { return '加载中…'; }
    if (!state.path) { return '等待数据：请在左侧选择案例或输入路径'; }
    return '暂无报告数据';
  }

  function loadNarrate(mockFile) {
    if (!state.path) { return; }
    /* 本轮任务约定：?mock=1 先读全局 /static/mock/narrate.json，404 再回退 <slug>.narrate.json */
    if (state.mock && !mockFile) { mockFile = 'narrate'; }
    var gen = state.dataGen;
    state.pending.narrate = true;
    state.failed.narrate = false;
    renderAll();
    var params = { path: state.path, provider: state.ui.narrateProvider || 'auto' };
    apiGet('narrate', params, mockFile).then(function (d) {
      if (gen !== state.dataGen) { return; }
      state.pending.narrate = false;
      state.failed.narrate = false;
      state.updatedAt = nowStamp();
      state.narrate = d;
      renderAll();
    }, function (err) {
      if (gen !== state.dataGen) { return; }
      /* mock 兜底：全局 narrate.json 不存在（404）时回退 <slug>.narrate.json */
      if (state.mock && mockFile === 'narrate' && err.status === 404) { loadNarrate(mockKeyFor(state.path) + '.narrate'); return; }
      state.pending.narrate = false;
      state.failed.narrate = true;
      recordError('narrate', err.status, err.message, err.url);
      renderAll();
    });
  }

  function narrateEvidenceCount(sections) {
    var n = 0;
    for (var i = 0; i < sections.length; i++) {
      var ev = (sections[i] && Array.isArray(sections[i].evidence)) ? sections[i].evidence : [];
      n += ev.length;
    }
    return n;
  }

  function renderNarrateTop() {
    if (!el.narrateMeta) { return; }
    var d = narrateData();
    if (!d) { el.narrateMeta.textContent = narrateEmptyText(); return; }
    var sections = Array.isArray(d.sections) ? d.sections : [];
    var bits = ['provider ' + (state.ui.narrateProvider || 'auto')];
    bits.push('sections ' + fmtInt(sections.length));
    bits.push('evidence ' + fmtInt(narrateEvidenceCount(sections)));
    if (isNum(d.confidence)) { bits.push('confidence ' + d.confidence.toFixed(2)); }
    el.narrateMeta.textContent = bits.join('  ·  ');
  }

  function renderNarrateHead() {
    var d = narrateData();
    if (el.narrateEmpty) {
      el.narrateEmpty.textContent = narrateEmptyText();
      show(el.narrateEmpty, !d);
    }
    if (el.narrateHeadline) {
      el.narrateHeadline.textContent = (d && d.headline) ? String(d.headline) : '—';
      el.narrateHeadline.className = 'headline' + (d ? '' : ' is-empty');
    }
    if (el.narrateMode) {
      var mode = (d && d.mode) ? String(d.mode) : null;
      if (mode === 'llm') { el.narrateMode.textContent = '模式 llm'; el.narrateMode.className = 'badge badge-accent'; }
      else if (mode === 'offline') { el.narrateMode.textContent = '模式 offline'; el.narrateMode.className = 'badge badge-na'; }
      else if (mode) { el.narrateMode.textContent = '模式 ' + mode; el.narrateMode.className = 'badge badge-mid'; }
      else { el.narrateMode.textContent = '模式 —'; el.narrateMode.className = 'badge badge-na'; }
    }
    if (el.narrateModel) {
      el.narrateModel.textContent = (d && d.model) ? ('模型 ' + String(d.model)) : '模型 —';
      el.narrateModel.className = 'badge ' + ((d && d.model) ? 'badge-accent' : 'badge-na');
    }
    if (el.narrateTime) {
      el.narrateTime.textContent = (d && d.generated_at) ? ('生成 ' + String(d.generated_at)) : '时间 —';
      el.narrateTime.className = 'badge badge-na';
    }
    if (el.narrateConfNum) {
      el.narrateConfNum.textContent = (d && isNum(d.confidence)) ? d.confidence.toFixed(2) : '—';
    }
    if (el.narrateConfFill) {
      var c = (d && isNum(d.confidence)) ? d.confidence : 0;
      if (c < 0) { c = 0; }
      if (c > 1) { c = 1; }
      el.narrateConfFill.style.width = (c * 100).toFixed(1) + '%';
      el.narrateConfFill.className = 'bar-fill' + (c < 0.4 ? ' is-neg' : '');
    }
  }

  function narrateEvidenceTable(evidence) {
    var ev = Array.isArray(evidence) ? evidence : [];
    if (!ev.length) { return mk('div', 'report-ev-empty', '无证据（evidence 为空）'); }
    var tbl = mk('table', 'ev-table');
    var thead = mk('thead');
    var hr = mk('tr');
    var hs = ['key', 'value', 'unit', 'source', 'confidence'];
    var i;
    for (i = 0; i < hs.length; i++) { hr.appendChild(mk('th', null, hs[i])); }
    thead.appendChild(hr);
    tbl.appendChild(thead);
    var tb = mk('tbody');
    for (i = 0; i < ev.length; i++) {
      var e = (ev[i] && typeof ev[i] === 'object') ? ev[i] : {};
      var tr = mk('tr');
      tr.appendChild(mk('td', 'k', (e.key !== undefined && e.key !== null && e.key !== '') ? String(e.key) : '—'));
      var vTxt;
      if (isNum(e.value)) { vTxt = String(e.value); }
      else if (typeof e.value === 'boolean') { vTxt = e.value ? 'true' : 'false'; }
      else if (typeof e.value === 'string' && e.value) { vTxt = e.value; }
      else { vTxt = '—'; }
      tr.appendChild(mk('td', 'num', vTxt));
      tr.appendChild(mk('td', null, e.unit ? String(e.unit) : '—'));
      tr.appendChild(mk('td', 'rate', e.source ? String(e.source) : '—'));
      var cc = isNum(e.confidence) ? e.confidence : null;
      var cTd = mk('td', 'num', cc === null ? '—' : cc.toFixed(2));
      if (cc !== null) { cTd.className += ' ev-conf-' + confClass(cc); }
      tr.appendChild(cTd);
      tb.appendChild(tr);
    }
    tbl.appendChild(tb);
    return tbl;
  }

  function renderNarrateSections() {
    var host = el.narrateSections;
    if (!host) { return; }
    clear(host);
    var d = narrateData();
    if (!d) { return; }
    var sections = Array.isArray(d.sections) ? d.sections : [];
    if (!sections.length) {
      host.appendChild(mk('div', 'empty', '报告没有 sections（sections 缺失或为空）'));
      return;
    }
    for (var i = 0; i < sections.length; i++) {
      var s = (sections[i] && typeof sections[i] === 'object') ? sections[i] : {};
      var card = mk('section', 'report-sec');
      var head = mk('div', 'report-sec-head');
      head.appendChild(mk('h3', 'report-sec-title', s.title ? String(s.title) : '(无标题)'));
      if (s.id) { head.appendChild(mk('span', 'report-sec-id', String(s.id))); }
      card.appendChild(head);
      if (s.text) { card.appendChild(mk('div', 'report-sec-text', String(s.text))); }
      card.appendChild(narrateEvidenceTable(s.evidence));
      host.appendChild(card);
    }
  }

  function renderNarrateList(host, countEl, list, kind, emptyMsg) {
    if (!host) { return; }
    clear(host);
    var d = narrateData();
    var arr = Array.isArray(list) ? list : [];
    if (countEl) { countEl.textContent = d ? (fmtInt(arr.length) + ' 条') : '—'; }
    if (!d) { host.appendChild(mk('div', 'empty', narrateEmptyText())); return; }
    if (!arr.length) { host.appendChild(mk('div', 'empty', emptyMsg)); return; }
    var ul = mk('ul', 'rep-list is-' + kind);
    for (var i = 0; i < arr.length; i++) { ul.appendChild(mk('li', null, String(arr[i]))); }
    host.appendChild(ul);
  }

  function renderNarrateLists() {
    var d = narrateData();
    renderNarrateList(el.narrateUncertainties, el.narrateUncCount, d && d.uncertainties, 'unc', 'uncertainties 为空');
    renderNarrateList(el.narrateActions, el.narrateActCount, d && d.next_actions, 'act', 'next_actions 为空');
  }

  function renderNarratePrompt() {
    if (el.narratePrompt) {
      var d = narrateData();
      var p = (d && typeof d.prompt_preview === 'string') ? d.prompt_preview : '';
      el.narratePrompt.textContent = p ? p : (d ? '（prompt_preview 为空）' : narrateEmptyText());
    }
    if (el.narratePromptFold) { show(el.narratePromptFold, true); }
  }

  function renderNarrateWarnings() {
    var host = el.narrateWarnings;
    if (!host) { return; }
    clear(host);
    var d = narrateData();
    if (!d) {
      setText('narrateWarnCount', '—');
      host.appendChild(mk('div', 'empty', narrateEmptyText()));
      return;
    }
    var list = Array.isArray(d.warnings) ? d.warnings : [];
    setText('narrateWarnCount', fmtInt(list.length) + ' 条');
    if (!list.length) { host.appendChild(mk('div', 'empty', '无告警')); return; }
    var ul = mk('ul', 'warnlist');
    for (var i = 0; i < list.length; i++) { ul.appendChild(mk('li', null, String(list[i]))); }
    host.appendChild(ul);
  }

  function renderNarrate() {
    renderNarrateTop();
    renderNarrateHead();
    renderNarrateLlmInfo();
    renderNarrateSections();
    renderNarrateLists();
    renderNarratePrompt();
    renderNarrateWarnings();
    /* provider 选择器随 LLM 配置状态给提示（未 ready 时标注会走离线渲染） */
    syncNarrateProviderHint();
  }

  function syncNarrateControls() {
    if (el.ctlNarrateProvider) { el.ctlNarrateProvider.value = state.ui.narrateProvider || 'auto'; }
  }

  /* ================= 设置页（LLM 配置，契约 §17） =================
   * 端点：GET /api/llm/config、POST /api/llm/config、POST /api/llm/test、GET /api/llm/models。
   * 硬约束：
   *   - setText(id, text) 第一参是 el 缓存的键名；本页一律显式用 el.xxx，不用字符串 key。
   *   - 绝不把密钥明文写进 textContent / title / 错误条 / console；输入框 type=password。
   *   - 进入本页只发 GET；页面加载不自动 POST；同一标签内不重复请求。
   *   - 渲染逐字段容错：字段缺失 / source 缺失 / models 为空 / 400 / 500 都不能崩。
   */
  var LLM_SOURCE_LABEL = { file: '本地文件', env: '环境变量', default: '默认' };
  var LLM_SOURCE_CLASS = { file: 'src-file', env: 'src-env', default: 'src-default' };
  var LLM_ENV_KEYS = ['SPXH_LLM_BASE_URL', 'SPXH_LLM_API_KEY', 'SPXH_LLM_MODEL'];

  function llmCfg() { return state.llm.config; }

  function llmSourceOf(key) {
    var c = llmCfg();
    var src = (c && c.source && typeof c.source === 'object') ? c.source[key] : null;
    return (typeof src === 'string' && LLM_SOURCE_LABEL[src]) ? src : '';
  }

  function llmReadyInfo(cfg) {
    var c = arguments.length ? cfg : llmCfg();
    if (!c || typeof c !== 'object') {
      return { known: false, ready: false, code: 'unknown', text: 'LLM 状态未知（尚未拿到 /api/llm/config）' };
    }
    var enabled = c.enabled !== false;
    var hasKey = c.api_key_set === true;
    var base = c.base_url ? String(c.base_url) : '';
    var model = c.model ? String(c.model) : '';
    if (c.ready === true) {
      return { known: true, ready: true, code: 'ready', text: 'LLM 已就绪' + (model ? ('（' + model + '）') : '') };
    }
    var reason;
    var code;
    if (!enabled) { reason = '配置中已关闭 LLM（enabled = false）'; code = 'disabled'; }
    else if (!hasKey) { reason = '未配置 API Key'; code = 'nokey'; }
    else if (!base) { reason = '未配置 Base URL'; code = 'nobase'; }
    else if (!model) { reason = '未配置模型 model'; code = 'nomodel'; }
    else { reason = '后端报告 ready=false'; code = 'other'; }
    return { known: true, ready: false, code: code, text: 'LLM 未就绪：' + reason };
  }

  function truncateText(s, n) {
    var t = String(s === null || s === undefined ? '' : s);
    var out = '';
    var prevSpace = false;
    for (var i = 0; i < t.length; i++) {
      var code = t.charCodeAt(i);
      var isWs = (code === 32 || code === 10 || code === 13 || code === 9);
      if (isWs) { if (!prevSpace) { out += ' '; } prevSpace = true; }
      else { out += t.charAt(i); prevSpace = false; }
    }
    return out.length > n ? (out.slice(0, n) + '…') : out;
  }

  function llmKvRow(host, k, v, cls, title) {
    var row = mk((host.tagName && host.tagName.toLowerCase() === 'ul') ? 'li' : 'div', 'kv-row' + (cls ? (' ' + cls) : ''));
    row.appendChild(mk('span', 'kv-k', String(k)));
    row.appendChild(mk('span', 'kv-v', String(v)));
    if (title) { row.title = String(title); }
    host.appendChild(row);
    return row;
  }

  function mkOption(value, label) {
    var o = document.createElement('option');
    o.value = value;
    o.textContent = label;
    return o;
  }

  /* ---------- 服务商预设（契约 §17 新增 presets / config_path_display） ---------- */

  /* 预设全部来自后端；缺失 / 非数组 / 元素非法一律跳过，不内置假数据、不猜。 */
  function llmPresetsList() {
    var c = llmCfg();
    var raw = (c && Array.isArray(c.presets)) ? c.presets : [];
    var out = [];
    for (var i = 0; i < raw.length; i++) {
      var p = raw[i];
      if (!p || typeof p !== 'object') { continue; }
      var label = (typeof p.label === 'string' && p.label) ? p.label : ((p.id !== undefined && p.id !== null) ? String(p.id) : '');
      if (!label) { continue; }
      out.push({
        id: (p.id !== undefined && p.id !== null) ? String(p.id) : ('preset-' + i),
        label: label,
        base_url: (typeof p.base_url === 'string') ? p.base_url : '',
        model: (typeof p.model === 'string') ? p.model : '',
        notes: (typeof p.notes === 'string') ? p.notes : ''
      });
    }
    return out;
  }

  function llmPresetSig(list) {
    var parts = [];
    for (var i = 0; i < list.length; i++) {
      parts.push([list[i].id, list[i].label, list[i].base_url, list[i].model, list[i].notes].join('\u0001'));
    }
    return parts.join('\u0002');
  }

  function llmPresetById(list, id) {
    for (var i = 0; i < list.length; i++) { if (list[i].id === id) { return list[i]; } }
    return null;
  }

  function llmFormBaseUrl() { return (el.llmBaseUrl && el.llmBaseUrl.value) ? String(el.llmBaseUrl.value).trim() : ''; }
  function llmFormModel() { return (el.llmModel && el.llmModel.value) ? String(el.llmModel.value).trim() : ''; }

  /* 两个 base_url 是否指向同一根地址：完全一致，或只差末尾的斜杠 / "/v1"。
     为什么要容忍 /v1：DeepSeek 预设给的是 https://api.deepseek.com，而用户常常保存成
     https://api.deepseek.com/v1（后端归一化不会去掉 /v1），两者都合法，不该一个高亮一个不高亮。 */
  function llmBaseUrlSame(a, b) {
    if (a === b) { return true; }
    var x = a, y = b;
    while (x.length && x.charAt(x.length - 1) === '/') { x = x.slice(0, -1); }
    while (y.length && y.charAt(y.length - 1) === '/') { y = y.slice(0, -1); }
    if (x === y) { return true; }
    if (x.length > 3 && x.slice(x.length - 3) === '/v1') { x = x.slice(0, x.length - 3); }
    if (y.length > 3 && y.slice(y.length - 3) === '/v1') { y = y.slice(0, y.length - 3); }
    return x === y;
  }

  /* 高亮判据：表单 base_url 与预设等价（见上）且 model 完全一致。
     custom 预设的 base_url / model 都是空串，所以两栏都空时高亮 custom。 */
  function llmPresetActiveId(list) {
    var base = llmFormBaseUrl();
    var model = llmFormModel();
    for (var i = 0; i < list.length; i++) {
      if (model === list[i].model && llmBaseUrlSame(base, list[i].base_url)) { return list[i].id; }
    }
    return '';
  }

  /* 一键填入：字面写入预设的 base_url / model（custom 预设两者为空串 = 清空两栏，
     便于重新输入，同时也让"空表单 = 自定义"的高亮自洽）；绝不碰 api_key、不提交、不改 baseline。 */
  function applyLlmPreset(p) {
    if (!p) { return; }
    if (el.llmBaseUrl) { el.llmBaseUrl.value = p.base_url; }
    if (el.llmModel) { el.llmModel.value = p.model; }
    state.llm.presetNotesId = p.id;
    renderLlmPresets();
  }

  function renderLlmPresets() {
    var list = llmPresetsList();
    var sig = llmPresetSig(list);
    var host = el.llmPresets;
    if (host && (sig !== state.llm.presetsSig || state.llm.presetNodes.length !== list.length)) {
      state.llm.presetsSig = sig;
      clear(host);
      state.llm.presetNodes = [];
      for (var i = 0; i < list.length; i++) {
        var p = list[i];
        var b = mk('button', 'btn btn-xs preset-btn', p.label);
        b.type = 'button';
        if (b.setAttribute) { b.setAttribute('data-preset-id', p.id); }
        if (p.notes) { b.title = p.notes; }
        b.addEventListener('click', (function (preset) {
          return function () { applyLlmPreset(preset); };
        })(p));
        host.appendChild(b);
        state.llm.presetNodes.push({ id: p.id, node: b });
      }
    }
    show(host, !!host && list.length > 0);
    var active = llmPresetActiveId(list);
    var nodes = state.llm.presetNodes;
    for (var j = 0; j < nodes.length; j++) {
      var on = active !== '' && nodes[j].id === active;
      nodes[j].node.className = 'btn btn-xs preset-btn' + (on ? ' is-active' : '');
      if (nodes[j].node.setAttribute) { nodes[j].node.setAttribute('aria-pressed', on ? 'true' : 'false'); }
    }
    var note = active ? llmPresetById(list, active) : (state.llm.presetNotesId ? llmPresetById(list, state.llm.presetNotesId) : null);
    if (el.llmPresetNotes) {
      if (note && note.notes) {
        el.llmPresetNotes.textContent = truncateText(note.notes, 72);
        el.llmPresetNotes.title = note.notes;
        show(el.llmPresetNotes, true);
      } else {
        el.llmPresetNotes.textContent = '';
        el.llmPresetNotes.title = '';
        show(el.llmPresetNotes, false);
      }
    }
  }

  /* config_path：优先显示相对路径 config_path_display（缺失回退 config_path），
     title 放绝对路径 config_path_abs（缺失回退 config_path），不谎报。 */
  function llmConfigPathInfo(c) {
    c = (c && typeof c === 'object') ? c : {};
    var abs = (typeof c.config_path_abs === 'string' && c.config_path_abs) ? c.config_path_abs : '';
    var display = (typeof c.config_path_display === 'string' && c.config_path_display) ? c.config_path_display : '';
    var raw = (typeof c.config_path === 'string' && c.config_path) ? c.config_path : '';
    return { display: display || raw, title: abs || raw, has: !!(display || raw || abs) };
  }

  /* ---------- 只读渲染 ---------- */

  function setLlmSourceBadge(key, sourceKey) {
    var n = el[key];
    if (!n) { return; }
    var s = llmSourceOf(sourceKey);
    if (s) {
      n.textContent = LLM_SOURCE_LABEL[s] + ' · ' + s;
      n.className = 'src-badge ' + (LLM_SOURCE_CLASS[s] || 'src-unknown');
      n.title = 'source: ' + s;
    } else {
      n.textContent = '来源 —';
      n.className = 'src-badge src-unknown';
      n.title = '响应未提供该项的 source';
    }
  }

  function renderLlmSources() {
    setLlmSourceBadge('llmSrcEnabled', 'enabled');
    setLlmSourceBadge('llmSrcBaseUrl', 'base_url');
    setLlmSourceBadge('llmSrcApiKey', 'api_key');
    setLlmSourceBadge('llmSrcModel', 'model');
    setLlmSourceBadge('llmSrcTemperature', 'temperature');
    setLlmSourceBadge('llmSrcMaxTokens', 'max_tokens');
    setLlmSourceBadge('llmSrcTimeout', 'timeout_s');
  }

  function renderLlmMeta() {
    if (!el.llmMeta) { return; }
    if (state.llm.loading) { el.llmMeta.textContent = '配置加载中…'; return; }
    if (state.llm.failed) { el.llmMeta.textContent = '配置加载失败：' + state.llm.error; return; }
    var c = llmCfg();
    if (!c) { el.llmMeta.textContent = '尚未加载配置'; return; }
    var bits = [];
    bits.push('enabled ' + (c.enabled !== false ? 'true' : 'false'));
    bits.push('ready ' + (c.ready === true ? 'true' : 'false'));
    bits.push('key ' + (c.api_key_set === true ? '已配置' : '未配置'));
    el.llmMeta.textContent = bits.join('  ·  ');
  }

  function renderLlmReady() {
    var info = llmReadyInfo();
    var cls = !info.known ? 'is-unknown' : (info.ready ? 'is-ok' : 'is-warn');
    if (el.llmReady) { el.llmReady.textContent = info.text; el.llmReady.className = 'llm-ready ' + cls; }
    if (el.llmReadyBanner) { el.llmReadyBanner.textContent = info.text; el.llmReadyBanner.className = 'llm-ready-banner ' + cls; }
    if (el.llmReadyChip) {
      el.llmReadyChip.textContent = !info.known ? '未知' : (info.ready ? '已就绪' : '未就绪');
      el.llmReadyChip.className = 'chip' + (info.ready ? ' chip-ok' : (info.known ? ' chip-warn' : ''));
      el.llmReadyChip.title = info.text;
    }
  }

  function renderLlmSummary() {
    var host = el.llmSummary;
    if (!host) { return; }
    clear(host);
    if (state.llm.loading) { host.appendChild(mk('div', 'empty', '配置加载中…')); return; }
    if (state.llm.failed) { host.appendChild(mk('div', 'llm-err', '配置加载失败：' + state.llm.error)); return; }
    var c = llmCfg();
    if (!c) { host.appendChild(mk('div', 'empty', '尚未加载配置')); return; }
    llmKvRow(host, 'ready', c.ready === true ? 'true' : 'false', c.ready === true ? 'is-ok' : 'is-warn');
    llmKvRow(host, 'enabled', c.enabled !== false ? 'true' : 'false', c.enabled !== false ? '' : 'is-warn');
    llmKvRow(host, 'base_url', c.base_url ? String(c.base_url) : '—');
    llmKvRow(host, 'model', c.model ? String(c.model) : '—');
    llmKvRow(host, 'temperature', isNum(c.temperature) ? String(c.temperature) : '—');
    llmKvRow(host, 'max_tokens', isNum(c.max_tokens) ? String(c.max_tokens) : '—');
    llmKvRow(host, 'timeout_s', isNum(c.timeout_s) ? String(c.timeout_s) : '—');
    var hint = (typeof c.api_key_hint === 'string' && c.api_key_hint) ? c.api_key_hint : null;
    llmKvRow(host, 'api_key', c.api_key_set === true ? ('已配置' + (hint ? ('（' + hint + '）') : '')) : '未配置',
      c.api_key_set === true ? 'is-ok' : 'is-warn');
    var pathInfo = llmConfigPathInfo(c);
    llmKvRow(host, 'config_path', pathInfo.has ? pathInfo.display : '—', undefined, pathInfo.has ? pathInfo.title : '');
  }

  function renderLlmEnv() {
    var host = el.llmEnv;
    if (!host) { return; }
    clear(host);
    var c = llmCfg();
    var env = (c && c.env && typeof c.env === 'object') ? c.env : null;
    for (var i = 0; i < LLM_ENV_KEYS.length; i++) {
      var key = LLM_ENV_KEYS[i];
      var v = env ? env[key] : null;
      var txt;
      if (key === 'SPXH_LLM_API_KEY') {
        /* 后端只报"是否设置"；即便它误回显了值，这里也只翻译成两态，永不上屏。 */
        txt = (v === '未设置' || v === false || v === null || v === undefined || v === '') ? '未设置' : '已设置';
      } else {
        txt = (typeof v === 'string' && v) ? v : ((v === undefined || v === null) ? '—' : String(v));
      }
      llmKvRow(host, key, txt, txt === '已设置' ? 'is-ok' : (txt === '未设置' ? 'is-warn' : ''));
    }
  }

  function renderLlmKeyState() {
    var c = llmCfg();
    var set = !!(c && c.api_key_set === true);
    var hint = (c && typeof c.api_key_hint === 'string' && c.api_key_hint) ? c.api_key_hint : null;
    if (el.llmKeyState) {
      var txt;
      if (state.llm.clearPending) { txt = '密钥 待清除（提交中…）'; }
      else if (set) { txt = '密钥 已配置' + (hint ? ('（' + hint + '）') : ''); }
      else { txt = '密钥 未配置'; }
      el.llmKeyState.textContent = txt;
      el.llmKeyState.className = 'mono key-state ' + ((set && !state.llm.clearPending) ? 'is-ok' : 'is-off');
    }
    if (el.llmKeyClear) { el.llmKeyClear.disabled = (!set) || state.llm.clearPending; }
  }

  function renderLlmModels() {
    var list = state.llm.models || [];
    var sig = list.join('|');
    if (sig !== state.llm.modelsSig) {
      state.llm.modelsSig = sig;
      if (el.llmModelSelect) {
        clear(el.llmModelSelect);
        el.llmModelSelect.appendChild(mkOption('', '（候选模型）'));
        for (var i = 0; i < list.length; i++) { el.llmModelSelect.appendChild(mkOption(String(list[i]), String(list[i]))); }
        el.llmModelSelect.value = '';
        show(el.llmModelSelect, list.length > 0);
      }
      if (el.llmModelOptions) {
        clear(el.llmModelOptions);
        for (var j = 0; j < list.length; j++) { el.llmModelOptions.appendChild(mkOption(String(list[j]), String(list[j]))); }
      }
    }
    if (el.llmModelsNote) {
      var note;
      if (state.pending.llmModels) { note = '拉取中…'; }
      else if (!state.llm.modelsLoaded) { note = '尚未拉取模型列表'; }
      else if (state.llm.modelsError) { note = '拉取失败：' + truncateText(state.llm.modelsError, 160); }
      else if (!list.length) { note = '接口返回 0 个候选模型（可继续手输模型名）'; }
      else { note = '共 ' + String(list.length) + ' 个候选模型' + (state.llm.modelsSource ? ('　source: ' + state.llm.modelsSource) : ''); }
      el.llmModelsNote.textContent = note;
    }
  }

  function renderLlmTest() {
    var t = state.llm.test;
    if (el.llmTestChip) {
      if (state.llm.testing) { el.llmTestChip.textContent = '测试中…'; el.llmTestChip.className = 'chip chip-warn'; }
      else if (state.llm.testError) { el.llmTestChip.textContent = '请求失败'; el.llmTestChip.className = 'chip chip-err'; }
      else if (!t) { el.llmTestChip.textContent = '未测试'; el.llmTestChip.className = 'chip'; }
      else { el.llmTestChip.textContent = (t.ok === true ? 'ok' : 'ok=false'); el.llmTestChip.className = 'chip ' + (t.ok === true ? 'chip-ok' : 'chip-err'); }
    }
    var host = el.llmTestResult;
    if (!host) { return; }
    clear(host);
    if (state.llm.testing) { host.appendChild(mk('div', 'empty', '测试中…（等待模型回应）')); return; }
    if (state.llm.testError) { host.appendChild(mk('div', 'llm-err', '测试请求失败：' + state.llm.testError)); return; }
    if (!t) { host.appendChild(mk('div', 'empty', '尚未测试')); return; }
    llmKvRow(host, 'ok', t.ok === true ? 'true' : 'false', t.ok === true ? 'is-ok' : 'is-err');
    llmKvRow(host, 'latency_ms', isNum(t.latency_ms) ? String(t.latency_ms) : '—');
    if (t.model) { llmKvRow(host, 'model', String(t.model)); }
    if (t.checked && typeof t.checked === 'object') {
      if (t.checked.base_url) { llmKvRow(host, 'checked.base_url', String(t.checked.base_url)); }
      if (t.checked.model) { llmKvRow(host, 'checked.model', String(t.checked.model)); }
      if (t.checked.api_key_hint) { llmKvRow(host, 'checked.api_key_hint', String(t.checked.api_key_hint)); }
      /* 实际请求地址：Base URL 写错（多写了 /chat/completions）时一眼看出来 */
      if (t.checked.chat_url) {
        llmKvRow(host, '实际请求地址', String(t.checked.chat_url), '', 'checked.chat_url');
      }
      if (t.checked.models_url) {
        llmKvRow(host, '模型列表地址', String(t.checked.models_url), '', 'checked.models_url');
      }
    }
    if (t.ok === true) {
      if (t.reply !== undefined && t.reply !== null && t.reply !== '') {
        host.appendChild(mk('div', 'llm-reply', String(t.reply)));
      } else {
        host.appendChild(mk('div', 'empty', 'reply 为空'));
      }
      var models = Array.isArray(t.models) ? t.models : [];
      if (models.length) {
        var head = models.slice(0, 8).join(', ');
        host.appendChild(mk('div', 'llm-models-line', 'models ' + String(models.length) + ' 个：' + head + (models.length > 8 ? ' …' : '')));
      }
    } else if (t.error) {
      host.appendChild(mk('div', 'llm-err', '错误原因：' + truncateText(t.error, 200)));
    } else {
      host.appendChild(mk('div', 'empty', 'ok=false 但未给出 error'));
    }
    /* hint：后端给的那句"为什么"（如"该服务商可用模型：…"）。可能缺失 / 空串，逐字段容错。 */
    if (typeof t.hint === 'string' && t.hint) {
      host.appendChild(mk('div', 'llm-test-hint', t.hint));
    }
  }

  function renderLlmSaveStatus() {
    if (!el.llmSaveStatus) { return; }
    var s = state.llm.saveState;
    var cls = 'llm-status';
    if (s === 'ok') { cls += ' is-ok'; }
    else if (s === 'err') { cls += ' is-err'; }
    else if (s === 'saving') { cls += ' is-busy'; }
    el.llmSaveStatus.className = cls;
    el.llmSaveStatus.textContent = state.llm.saveMessage || '—';
  }

  function renderLlmConfigPath() {
    if (!el.llmConfigPath) { return; }
    var c = llmCfg();
    if (state.llm.loading) { el.llmConfigPath.textContent = 'config_path 加载中…'; el.llmConfigPath.title = ''; return; }
    if (state.llm.failed) { el.llmConfigPath.textContent = 'config_path 未知'; el.llmConfigPath.title = ''; return; }
    var info = llmConfigPathInfo(c);
    if (!info.has) { el.llmConfigPath.textContent = 'config_path —'; el.llmConfigPath.title = ''; return; }
    /* 正文用相对路径（config_path_display 优先），title 放绝对路径 */
    el.llmConfigPath.textContent = 'config_path ' + info.display;
    el.llmConfigPath.title = info.title;
  }

  function renderLlmSettings() {
    renderLlmReady();
    renderLlmConfigPath();
    renderLlmPresets();
    renderLlmSources();
    renderLlmMeta();
    renderLlmSummary();
    renderLlmEnv();
    renderLlmKeyState();
    renderLlmModels();
    renderLlmTest();
    renderLlmSaveStatus();
  }

  /* ---------- 表单 / 请求 ---------- */

  function llmNumOrNull(v) {
    if (v === null || v === undefined || v === '') { return null; }
    var n = Number(v);
    return isFinite(n) ? n : null;
  }

  function collectLlmForm() {
    return {
      enabled: !!(el.llmEnabled && el.llmEnabled.checked),
      base_url: (el.llmBaseUrl && el.llmBaseUrl.value) ? String(el.llmBaseUrl.value).trim() : '',
      model: (el.llmModel && el.llmModel.value) ? String(el.llmModel.value).trim() : '',
      temperature: el.llmTemperature ? llmNumOrNull(el.llmTemperature.value) : null,
      max_tokens: el.llmMaxTokens ? llmNumOrNull(el.llmMaxTokens.value) : null,
      timeout_s: el.llmTimeout ? llmNumOrNull(el.llmTimeout.value) : null
    };
  }

  function llmBaselineFromConfig(c) {
    c = (c && typeof c === 'object') ? c : {};
    return {
      enabled: c.enabled !== false,
      base_url: c.base_url ? String(c.base_url) : '',
      model: c.model ? String(c.model) : '',
      temperature: isNum(c.temperature) ? c.temperature : null,
      max_tokens: isNum(c.max_tokens) ? c.max_tokens : null,
      timeout_s: isNum(c.timeout_s) ? c.timeout_s : null
    };
  }

  function applyLlmConfig(c) {
    state.llm.config = (c && typeof c === 'object') ? c : null;
    state.llm.baseline = state.llm.config ? llmBaselineFromConfig(state.llm.config) : null;
    state.llm.failed = false;
    state.llm.error = '';
    var cfg = state.llm.config || {};
    if (el.llmEnabled) { el.llmEnabled.checked = cfg.enabled !== false; }
    if (el.llmBaseUrl) { el.llmBaseUrl.value = cfg.base_url ? String(cfg.base_url) : ''; }
    if (el.llmModel) { el.llmModel.value = cfg.model ? String(cfg.model) : ''; }
    var tv = isNum(cfg.temperature) ? cfg.temperature : 0.2;
    if (el.llmTemperature) { el.llmTemperature.value = String(tv); }
    if (el.llmTemperatureRange) { el.llmTemperatureRange.value = String(tv); }
    if (el.llmMaxTokens) { el.llmMaxTokens.value = isNum(cfg.max_tokens) ? String(cfg.max_tokens) : ''; }
    if (el.llmTimeout) { el.llmTimeout.value = isNum(cfg.timeout_s) ? String(cfg.timeout_s) : ''; }
    /* 服务端永不回显密钥明文：表单里也只清空，不填任何东西。 */
    if (el.llmApiKey) { el.llmApiKey.value = ''; }
    state.llm.apiKeyDraft = '';
    /* 新配置落地：清掉"上次点击"的预设备注，高亮与 notes 完全由当前表单值决定 */
    state.llm.presetNotesId = '';
    setLlmKeyVisible(false);
  }

  function setLlmKeyVisible(on) {
    state.llm.showKey = !!on;
    if (el.llmApiKey) {
      try { el.llmApiKey.type = on ? 'text' : 'password'; } catch (e1) { /* 某些实现只读 */ }
      if (el.llmApiKey.setAttribute) { el.llmApiKey.setAttribute('type', on ? 'text' : 'password'); }
    }
    if (el.llmKeyEye) {
      el.llmKeyEye.textContent = on ? '隐藏' : '显示';
      if (el.llmKeyEye.setAttribute) { el.llmKeyEye.setAttribute('aria-pressed', on ? 'true' : 'false'); }
      el.llmKeyEye.title = on ? '恢复为密码框' : '显示输入内容（明文只在本输入框内，不写入页面文本）';
    }
  }

  function loadLlmConfig() {
    if (state.llm.loading) { return; }
    state.llm.loading = true;
    state.llm.failed = false;
    state.pending.llm = true;
    renderAll();
    apiGet('llm/config', {}, 'llm_config').then(function (d) {
      state.llm.loading = false;
      state.pending.llm = false;
      applyLlmConfig(d);
      renderAll();
    }, function (err) {
      state.llm.loading = false;
      state.pending.llm = false;
      state.llm.failed = true;
      state.llm.error = err.message;
      state.llm.config = null;
      state.llm.baseline = null;
      recordError('llm/config', err.status, err.message, err.url);
      renderAll();
    });
  }

  function ensureLlmConfig() {
    if (state.llm.config || state.llm.loading || state.llm.failed) { return; }
    loadLlmConfig();
  }

  function loadLlmModels() {
    if (state.pending.llmModels) { return; }
    state.pending.llmModels = true;
    state.llm.modelsError = null;
    renderAll();
    apiGet('llm/models', {}, 'llm_models').then(function (d) {
      state.pending.llmModels = false;
      state.llm.modelsLoaded = true;
      state.llm.models = (d && Array.isArray(d.models)) ? d.models.slice() : [];
      state.llm.modelsError = (d && d.error) ? String(d.error) : null;
      state.llm.modelsSource = (d && d.source) ? String(d.source) : '';
      renderAll();
    }, function (err) {
      state.pending.llmModels = false;
      state.llm.modelsLoaded = true;
      state.llm.models = [];
      state.llm.modelsError = err.message;
      state.llm.modelsSource = '';
      recordError('llm/models', err.status, err.message, err.url);
      renderAll();
    });
  }

  function mergeLlmModels(models) {
    if (!Array.isArray(models) || !models.length) { return; }
    var seen = {};
    var out = [];
    var i;
    for (i = 0; i < state.llm.models.length; i++) {
      var a = String(state.llm.models[i]);
      if (!seen[a]) { seen[a] = true; out.push(a); }
    }
    for (i = 0; i < models.length; i++) {
      var b = String(models[i]);
      if (!seen[b]) { seen[b] = true; out.push(b); }
    }
    state.llm.models = out;
    state.llm.modelsLoaded = true;
    state.llm.modelsError = null;
  }

  function testLlm() {
    if (state.llm.testing || state.pending.llmTest) { return; }
    var body = {};
    if (el.llmTestOverride && el.llmTestOverride.checked) {
      var f = collectLlmForm();
      body.base_url = f.base_url;
      body.model = f.model;
      /* 只有用户真输了新密钥才带上去；否则用后端已保存的密钥。 */
      if (state.llm.apiKeyDraft) { body.api_key = state.llm.apiKeyDraft; }
    }
    state.llm.testing = true;
    state.pending.llmTest = true;
    state.llm.test = null;
    state.llm.testError = '';
    renderAll();
    apiPost(API_PREFIX + '/llm/test', body, { api: 'llm/test', timeoutMs: 60000 }).then(function (d) {
      state.llm.testing = false;
      state.pending.llmTest = false;
      state.llm.test = (d && typeof d === 'object') ? d : null;
      if (d && Array.isArray(d.models) && d.models.length) { mergeLlmModels(d.models); }
      renderAll();
    }, function (err) {
      state.llm.testing = false;
      state.pending.llmTest = false;
      state.llm.test = null;
      state.llm.testError = err.message;
      recordError('llm/test', err.status, err.message, err.url);
      renderAll();
    });
  }

  function llmDiff() {
    var cur = collectLlmForm();
    var base = state.llm.baseline;
    var body = {};
    if (!base) {
      body.enabled = cur.enabled;
      body.base_url = cur.base_url;
      body.model = cur.model;
      if (cur.temperature !== null) { body.temperature = cur.temperature; }
      if (cur.max_tokens !== null) { body.max_tokens = cur.max_tokens; }
      if (cur.timeout_s !== null) { body.timeout_s = cur.timeout_s; }
      return body;
    }
    if (cur.enabled !== base.enabled) { body.enabled = cur.enabled; }
    if (cur.base_url !== base.base_url) { body.base_url = cur.base_url; }
    if (cur.model !== base.model) { body.model = cur.model; }
    if (cur.temperature !== base.temperature && cur.temperature !== null) { body.temperature = cur.temperature; }
    if (cur.max_tokens !== base.max_tokens && cur.max_tokens !== null) { body.max_tokens = cur.max_tokens; }
    if (cur.timeout_s !== base.timeout_s && cur.timeout_s !== null) { body.timeout_s = cur.timeout_s; }
    return body;
  }

  function saveLlmConfig() {
    if (state.pending.llmSave) { return; }
    var body = llmDiff();
    /* 关键：只有用户真的输入了新密钥才提交 api_key；留空绝不提交，
       否则会用空串覆盖已保存的密钥（这是本页最容易踩的坑）。 */
    if (state.llm.apiKeyDraft) { body.api_key = state.llm.apiKeyDraft; }
    if (state.llm.clearPending) { body.clear_api_key = true; }
    if (!hasKeys(body)) {
      state.llm.saveState = 'noop';
      state.llm.saveMessage = '没有需要保存的改动（未提交 api_key，避免覆盖已保存的密钥）';
      renderLlmSettings();
      return;
    }
    /* 记下用户提交的原值：保存成功后用响应值回填，若被后端归一化就明确告知 */
    var submittedBaseUrl = (typeof body.base_url === 'string') ? body.base_url : '';
    state.pending.llmSave = true;
    state.llm.saveState = 'saving';
    state.llm.saveMessage = '保存中…';
    renderAll();
    apiPost(API_PREFIX + '/llm/config', body, { api: 'llm/config' }).then(function (d) {
      state.pending.llmSave = false;
      state.llm.clearPending = false;
      /* 用响应值整体回填表单 + baseline（而不是沿用用户输入），归一化后的地址才会体现出来 */
      applyLlmConfig(d);
      state.llm.saveState = 'ok';
      var info = llmReadyInfo(d);
      var pathInfo = llmConfigPathInfo(d);
      var savedBase = (d && typeof d.base_url === 'string') ? d.base_url : '';
      var normNote = (submittedBaseUrl && savedBase && submittedBaseUrl !== savedBase) ? (' · base_url 已归一化为 ' + savedBase) : '';
      state.llm.saveMessage = '已保存 · ' + info.text + normNote + (pathInfo.has ? (' · ' + pathInfo.display) : '');
      renderAll();
    }, function (err) {
      state.pending.llmSave = false;
      state.llm.clearPending = false;
      state.llm.saveState = 'err';
      state.llm.saveMessage = '保存失败：' + err.message;
      recordError('llm/config', err.status, err.message, err.url);
      renderAll();
    });
  }

  function confirmDialog(message) {
    try {
      if (typeof window !== 'undefined' && typeof window.confirm === 'function') { return window.confirm(message) === true; }
      if (typeof confirm === 'function') { return confirm(message) === true; }
    } catch (e2) { /* some environments disable confirm */ }
    return false;
  }

  function clearLlmKey() {
    var c = llmCfg();
    if (!c || c.api_key_set !== true) {
      state.llm.saveState = 'noop';
      state.llm.saveMessage = '当前没有已保存的密钥可清除';
      renderLlmSettings();
      return;
    }
    var ok = confirmDialog('确定清除已保存的 LLM API Key 吗？清除后 LLM 将不可用（/api/narrate 回退离线渲染），直到重新配置。此操作不可撤销。');
    if (!ok) {
      state.llm.saveState = 'noop';
      state.llm.saveMessage = '已取消清除密钥（未提交 clear_api_key）';
      renderLlmSettings();
      return;
    }
    state.llm.clearPending = true;
    saveLlmConfig();
  }

  /* ---------- 报告页联动 ---------- */

  function renderNarrateLlmInfo() {
    if (!el.narrateLlmInfo) { return; }
    var d = narrateData();
    var info = (d && d.llm && typeof d.llm === 'object') ? d.llm : null;
    if (!info) {
      var mode = (d && d.mode) ? String(d.mode) : '';
      el.narrateLlmInfo.textContent = mode === 'offline' ? 'LLM 未参与（离线）' : 'LLM —';
      el.narrateLlmInfo.className = 'badge badge-na';
      el.narrateLlmInfo.title = '响应未带 llm 字段（契约 §17.5：仅当走 LLM 时非空）';
      return;
    }
    var bits = [];
    if (info.model) { bits.push('模型 ' + String(info.model)); }
    if (info.base_url) { bits.push(String(info.base_url)); }
    if (info.api_key_hint) { bits.push('密钥 ' + String(info.api_key_hint)); }
    el.narrateLlmInfo.textContent = bits.length ? ('LLM ' + bits.join(' · ')) : 'LLM 已使用';
    el.narrateLlmInfo.className = 'badge badge-accent';
    el.narrateLlmInfo.title = '来自 /api/narrate 的 llm 字段（脱敏）';
  }

  function syncNarrateProviderHint() {
    var c = llmCfg();
    var info = llmReadyInfo(c);
    var hint = '';
    if (!c) { hint = 'LLM 配置状态未知（打开「设置」查看）'; }
    else if (!info.ready) { hint = '未配置 LLM，将使用离线渲染（' + info.text + '）'; }
    if (el.narrateLlmHint) {
      el.narrateLlmHint.textContent = hint;
      show(el.narrateLlmHint, !!hint);
    }
    var sel = el.ctlNarrateProvider;
    if (sel && sel.options && sel.options.length) {
      for (var i = 0; i < sel.options.length; i++) {
        var o = sel.options[i];
        var v = o && o.value ? String(o.value) : '';
        if (v === 'auto' || v === 'llm') {
          if (c && !info.ready) {
            o.title = '未配置 LLM，将使用离线渲染';
            o.className = 'opt-warn';
          } else {
            o.title = '';
            o.className = '';
          }
        }
      }
    }
    if (sel) {
      sel.title = (c && !info.ready) ? '未配置 LLM，将使用离线渲染' : '';
    }
  }

  /* ================= 访问令牌（契约 §19） =================
   * 端点：GET /api/auth/status。
   * 硬约束：
   *   - 令牌只存在 localStorage + 内存 + password 输入框；
   *     绝不写进 textContent / title / placeholder / 错误信息 / 请求 URL / console。
   *   - 已保存令牌 -> 所有 apiGet/apiPost 自动带 Authorization: Bearer <token>；留空则不带。
   *   - 收到 401 时错误条给出固定中文提示（apiCore 里统一处理），不静默。
   */

  function readStoredToken() {
    try {
      if (typeof localStorage === 'undefined' || !localStorage) { return ''; }
      var raw = localStorage.getItem(AUTH_TOKEN_KEY);
      return (typeof raw === 'string' && raw) ? raw : '';
    } catch (err) {
      /* 隐私模式 / localStorage 被禁用：降级为"无令牌"，不崩、不提示明文 */
      return '';
    }
  }

  function writeStoredToken(token) {
    try {
      if (typeof localStorage === 'undefined' || !localStorage) { return false; }
      if (token) { localStorage.setItem(AUTH_TOKEN_KEY, token); }
      else { localStorage.removeItem(AUTH_TOKEN_KEY); }
      return true;
    } catch (err) {
      return false;
    }
  }

  function saveAuthToken() {
    var v = (el.authToken && el.authToken.value) ? String(el.authToken.value) : '';
    /* 粘贴常带首尾空白/换行；令牌本身不允许空白 */
    v = v.replace(/\s+/g, '');
    state.auth.token = v;
    state.auth.persistFailed = (v !== '' && !writeStoredToken(v));
    /* 保存后立刻清空输入框：令牌不驻留 DOM（仅存在于内存与 localStorage） */
    if (el.authToken) { el.authToken.value = ''; }
    renderAuthStatus();
  }

  function clearAuthToken() {
    state.auth.token = '';
    state.auth.persistFailed = false;
    writeStoredToken('');
    if (el.authToken) { el.authToken.value = ''; }
    renderAuthStatus();
  }

  function loadAuthStatus() {
    if (state.pending.auth) { return; }
    state.pending.auth = true;
    state.auth.failed = false;
    renderAll();
    apiGet('auth/status', {}, 'auth_status').then(function (d) {
      state.pending.auth = false;
      state.auth.failed = false;
      state.auth.status = (d && typeof d === 'object') ? d : {};
      renderAll();
    }, function (err) {
      state.pending.auth = false;
      state.auth.failed = true;
      state.auth.status = null;
      state.auth.error = err.message;
      recordError('auth/status', err.status, err.message, err.url);
      renderAll();
    });
  }

  /* 只发一次 GET：已有状态 / 在途 / 已失败（等用户点「刷新状态」）都不重复请求 */
  function ensureAuthStatus() {
    if (state.auth.status || state.pending.auth || state.auth.failed) { return; }
    loadAuthStatus();
  }

  function renderAuthStatus() {
    var st = state.auth.status;
    var hasToken = !!(state.auth && state.auth.token);
    if (el.authState) {
      var cls = 'chip mono';
      var txt = '状态未知';
      if (state.pending.auth) { txt = '读取中…'; cls += ' chip-warn'; }
      else if (state.auth.failed) { txt = '状态不可用'; cls += ' chip-err'; }
      else if (st && st.auth_required === true) { txt = '需要令牌'; cls += ' chip-warn'; }
      else if (st) { txt = '无需令牌'; cls += ' chip-ok'; }
      el.authState.textContent = txt;
      el.authState.className = cls;
    }
    if (el.authHint) {
      var hint;
      if (state.pending.auth) { hint = '正在读取 /api/auth/status…'; }
      else if (state.auth.failed) { hint = '读取 /api/auth/status 失败（见顶部错误条）：' + truncateText(state.auth.error || '', 160); }
      else if (!st) { hint = '尚未读取访问令牌状态（进入本页会自动读取一次）'; }
      else {
        var bits = [];
        bits.push(st.auth_required === true ? '当前接口需要访问令牌' : '当前接口不需要访问令牌');
        bits.push(st.loopback_only === true ? '仅监听回环地址' : '可被非本机访问');
        if (typeof st.hint === 'string' && st.hint) { bits.push(st.hint); }
        if (st.auth_required === true && !hasToken) { bits.push('未保存令牌：请求会得到 401，请在下方填入令牌后重试'); }
        else if (st.auth_required !== true && hasToken) { bits.push('已保存令牌：请求会带上它（服务器当前不校验）'); }
        hint = bits.join(' · ');
      }
      el.authHint.textContent = hint;
      el.authHint.className = 'hint' + ((st && st.auth_required === true && !hasToken) ? ' auth-hint-warn' : '');
    }
    if (el.authTokenState) {
      var ts;
      var tcls;
      if (hasToken) { ts = '令牌 已保存（长度 ' + String(state.auth.token.length) + '，仅本机）'; tcls = 'mono key-state is-ok'; }
      else { ts = '令牌 未保存'; tcls = 'mono key-state is-off'; }
      if (state.auth.persistFailed) { ts += ' · 浏览器拒绝写入 localStorage（仅本次会话有效）'; tcls = 'mono key-state is-warn'; }
      el.authTokenState.textContent = ts;
      el.authTokenState.className = tcls;
    }
    if (el.authTokenClear) { el.authTokenClear.disabled = !hasToken; }
  }

  /* ================= 代理页（契约 §18：工具调用代理） =================
   * 端点：GET /api/agent/tools（工具清单 + budgets）、POST /api/agent/run（一次运行）。
   * 硬约束：
   *   - 进入本页只发 GET，绝不自动运行；运行必须由用户点击触发。
   *   - 渲染逐字段容错：rounds 空 / tool_calls 空 / final 缺失 / stop_reason 未知 /
   *     grounding 缺失 / warnings 缺失都不能崩，也不能停在"加载中…"。
   *   - 失败态固定文案「代理运行失败（见顶部错误条）」。
   */
  var AGENT_STOP_TEXT = {
    final: 'final：模型自己认为已经收尾，正常结束',
    max_rounds: 'max_rounds：轮次预算用尽，被迫停止（结论可能还没收敛）',
    max_tool_calls: 'max_tool_calls：工具调用预算用尽，被迫停止',
    timeout: 'timeout：墙钟超时，循环被中断（已完成的轮次仍然有效）',
    llm_error: 'llm_error：LLM 调用出错，循环提前终止',
    no_llm: 'no_llm：没有可用 LLM，走了离线确定性计划（analyze → classify → frame → ssca）'
  };
  var AGENT_DEFAULT_BUDGET = { max_rounds: 6, max_tool_calls: 12, timeout_s: 120 };

  function agentData() { return state.agent.run; }

  function agentBudget() {
    if (state.agent.budgets && typeof state.agent.budgets === 'object') { return state.agent.budgets; }
    var d = agentData();
    if (d && d.trace_summary && d.trace_summary.budget && typeof d.trace_summary.budget === 'object') { return d.trace_summary.budget; }
    return null;
  }

  function agentBudgetTimeout() {
    var b = agentBudget();
    if (b && isNum(b.timeout_s) && b.timeout_s > 0) { return b.timeout_s; }
    return AGENT_DEFAULT_BUDGET.timeout_s;
  }

  function agentMaxRounds() {
    var b = agentBudget();
    if (b && isNum(b.max_rounds) && b.max_rounds > 0) { return Math.min(12, Math.round(b.max_rounds)); }
    return AGENT_DEFAULT_BUDGET.max_rounds;
  }

  function agentProviderValue() {
    var v = (el.ctlAgentProvider && el.ctlAgentProvider.value) ? String(el.ctlAgentProvider.value) : '';
    return (v === 'offline' || v === 'auto') ? v : 'auto';
  }

  function agentRoundsValue() {
    var raw = (el.ctlAgentRounds && el.ctlAgentRounds.value !== undefined && el.ctlAgentRounds.value !== '') ? parseInt(el.ctlAgentRounds.value, 10) : NaN;
    var v = (isFinite(raw) && raw >= 1) ? Math.round(raw) : agentMaxRounds();
    if (v < 1) { v = 1; }
    if (v > 12) { v = 12; }   /* 契约 §18：max_rounds 上限 12 */
    return v;
  }

  function fmtElapsed(ms) {
    if (!isNum(ms) || ms < 0) { return '—'; }
    if (ms < 1000) { return String(Math.round(ms)) + ' ms'; }
    return (ms / 1000).toFixed(1) + ' s';
  }

  function agentStopClass(reason) {
    if (reason === 'final') { return 'badge badge-hi'; }
    if (reason === 'max_rounds' || reason === 'max_tool_calls' || reason === 'timeout') { return 'badge badge-mid'; }
    if (reason === 'llm_error') { return 'badge badge-lo'; }
    return 'badge badge-na';
  }

  function agentStopSuffix(reason) {
    if (reason === 'final') { return ' is-ok'; }
    if (reason === 'max_rounds' || reason === 'max_tool_calls' || reason === 'timeout') { return ' is-warn'; }
    if (reason === 'llm_error') { return ' is-err'; }
    return '';
  }

  function agentStopText(reason) {
    if (reason === undefined || reason === null || reason === '') { return 'stop_reason 缺失：后端未给出停止原因'; }
    var t = AGENT_STOP_TEXT[String(reason)];
    return t ? t : ('未知 stop_reason：' + String(reason) + '（前端原样显示，不猜语义）');
  }

  function agentProgressText() {
    var b = agentBudget();
    var parts = ['运行中… 已等待 ' + fmtElapsed(state.agent.elapsedMs)];
    if (b && isNum(b.max_rounds)) { parts.push('最多 ' + fmtInt(b.max_rounds) + ' 轮'); }
    if (b && isNum(b.max_tool_calls)) { parts.push('最多 ' + fmtInt(b.max_tool_calls) + ' 次工具调用'); }
    if (b && isNum(b.timeout_s)) { parts.push('墙钟 ' + fmtInt(b.timeout_s) + ' 秒'); }
    return parts.join(' · ');
  }

  function agentEmptyText() {
    if (state.agent.running) { return agentProgressText(); }
    if (state.agent.failed) { return '代理运行失败（见顶部错误条）'; }
    if (!state.path) { return '等待数据：请在左侧选择案例或输入路径'; }
    return '尚未运行：填写目标（可空）后点「运行代理」；进入本页不会自动运行。';
  }

  function agentCountCalls(d) {
    if (!d || !Array.isArray(d.rounds)) { return null; }
    var n = 0;
    for (var i = 0; i < d.rounds.length; i++) {
      var r = d.rounds[i];
      if (r && Array.isArray(r.tool_calls)) { n += r.tool_calls.length; }
    }
    return n;
  }

  function agentArgsText(args) {
    if (args === undefined || args === null) { return '—'; }
    if (typeof args !== 'object') { return truncateText(String(args), 160); }
    var parts = [];
    for (var k in args) {
      if (!Object.prototype.hasOwnProperty.call(args, k)) { continue; }
      var v = args[k];
      var t;
      if (typeof v === 'string') { t = v; }
      else if (v === null || v === undefined) { t = 'null'; }
      else { try { t = JSON.stringify(v); } catch (err) { t = String(v); } }
      parts.push(k + '=' + t);
    }
    if (!parts.length) { return '（无参数）'; }
    return truncateText(parts.join('  '), 160);
  }

  function loadAgentTools() {
    if (state.pending.agentTools) { return; }
    state.pending.agentTools = true;
    state.agent.toolsFailed = false;
    renderAll();
    apiGet('agent/tools', {}, 'agent_tools').then(function (d) {
      state.pending.agentTools = false;
      state.agent.toolsFailed = false;
      state.agent.tools = (d && typeof d === 'object') ? d : {};
      state.agent.budgets = (d && d.budgets && typeof d.budgets === 'object') ? d.budgets : null;
      syncAgentControls();
      renderAll();
    }, function (err) {
      state.pending.agentTools = false;
      state.agent.toolsFailed = true;
      state.agent.tools = null;
      state.agent.budgets = null;
      recordError('agent/tools', err.status, err.message, err.url);
      renderAll();
    });
  }

  function ensureAgentTools() {
    if (state.agent.tools || state.pending.agentTools || state.agent.toolsFailed) { return; }
    loadAgentTools();
  }

  function agentStopTicker() {
    if (state.agent.timer) {
      clearInterval(state.agent.timer);
      state.agent.timer = 0;
    }
  }

  function agentStartTicker() {
    agentStopTicker();
    state.agent.timer = setInterval(function () {
      if (!state.agent.running) { return; }
      state.agent.elapsedMs = Date.now() - state.agent.startedAt;
      renderAgentStatusOnly();
    }, 500);
  }

  function runAgent() {
    if (state.agent.running) { return; }
    if (!state.path) {
      recordError('客户端', 0, '未选择数据路径：代理需要一个信号文件才能开始', '');
      renderAll();
      return;
    }
    var body = {
      path: state.path,
      goal: (el.agentGoal && el.agentGoal.value) ? String(el.agentGoal.value).trim() : '',
      max_rounds: agentRoundsValue(),
      provider: agentProviderValue()
    };
    state.agent.gen++;
    var gen = state.agent.gen;
    state.agent.running = true;
    state.agent.failed = false;
    state.agent.error = '';
    state.agent.run = null;
    state.agent.runPath = state.path;
    state.agent.startedAt = Date.now();
    state.agent.elapsedMs = 0;
    agentStartTicker();
    renderAll();
    /* 单发 POST（契约 §18 没有流式/轮询端点）：超时按 budgets.timeout_s + 15 s 余量给，
       上限与后端一致（300 s），避免请求先于后端超时而客户端自己放弃。 */
    var budgetS = agentBudgetTimeout();
    var timeoutMs = Math.min(300 + 15, budgetS + 15) * 1000;
    apiPost(API_PREFIX + '/agent/run', body, { api: 'agent/run', timeoutMs: timeoutMs }).then(function (d) {
      if (gen !== state.agent.gen) { return; }
      state.agent.running = false;
      state.agent.elapsedMs = Date.now() - state.agent.startedAt;
      agentStopTicker();
      state.agent.run = (d && typeof d === 'object') ? d : {};
      state.agent.failed = false;
      state.agent.updatedAt = nowStamp();
      state.updatedAt = nowStamp();
      renderAll();
    }, function (err) {
      if (gen !== state.agent.gen) { return; }
      state.agent.running = false;
      state.agent.elapsedMs = Date.now() - state.agent.startedAt;
      agentStopTicker();
      state.agent.run = null;
      state.agent.failed = true;
      state.agent.error = err.message;
      recordError('agent/run', err.status, err.message, err.url);
      renderAll();
    });
  }

  function renderAgentBudget() {
    if (!el.agentBudget) { return; }
    if (state.pending.agentTools) { el.agentBudget.textContent = '预算 读取中…'; el.agentBudget.title = ''; return; }
    var b = agentBudget();
    var tools = (state.agent.tools && Array.isArray(state.agent.tools.tools)) ? state.agent.tools.tools : [];
    if (!b) {
      el.agentBudget.textContent = state.agent.toolsFailed ? '预算 不可用（/api/agent/tools 失败）' : '预算 —';
      el.agentBudget.title = state.agent.toolsFailed ? 'GET /api/agent/tools 未成功；仍可直接运行代理' : '';
      return;
    }
    var parts = [];
    if (isNum(b.max_rounds)) { parts.push('≤' + fmtInt(b.max_rounds) + ' 轮'); }
    if (isNum(b.max_tool_calls)) { parts.push('≤' + fmtInt(b.max_tool_calls) + ' 次调用'); }
    if (isNum(b.timeout_s)) { parts.push('≤' + fmtInt(b.timeout_s) + ' s'); }
    if (tools.length) { parts.push(tools.length + ' 个工具'); }
    el.agentBudget.textContent = '预算 ' + (parts.length ? parts.join(' / ') : '—');
    var names = [];
    for (var i = 0; i < tools.length; i++) {
      var t = tools[i];
      if (t && t.name) { names.push(String(t.name)); }
    }
    el.agentBudget.title = names.length ? names.join(', ') : '';
  }

  function renderAgentStatusOnly() {
    if (el.agentStatus && state.agent.running) {
      el.agentStatus.textContent = agentProgressText();
      el.agentStatus.className = 'agent-status mono is-busy';
    }
    if (el.agentEmpty && state.agent.running) { el.agentEmpty.textContent = agentEmptyText(); }
  }

  function renderAgentStatus() {
    if (el.agentRun) { el.agentRun.disabled = !!state.agent.running; }
    if (el.agentStatus) {
      var cls = 'agent-status mono';
      var txt;
      if (state.agent.running) { txt = agentProgressText(); cls += ' is-busy'; }
      else if (state.agent.failed) { txt = '运行失败（见顶部错误条）'; cls += ' is-err'; }
      else if (agentData()) { txt = '已完成 · ' + fmtElapsed(state.agent.elapsedMs) + (state.agent.updatedAt ? (' · ' + state.agent.updatedAt) : ''); cls += ' is-ok'; }
      else if (!state.path) { txt = '未选择数据路径'; }
      else { txt = '尚未运行'; }
      el.agentStatus.textContent = txt;
      el.agentStatus.className = cls;
    }
    if (el.agentEmpty) {
      el.agentEmpty.textContent = agentEmptyText();
      show(el.agentEmpty, !agentData());
    }
  }

  function renderAgentHeader() {
    var d = agentData();
    var f = (d && d.final && typeof d.final === 'object') ? d.final : null;
    if (el.agentMode) {
      var mode = (d && d.mode) ? String(d.mode) : null;
      if (mode === 'llm') { el.agentMode.textContent = '模式 llm'; el.agentMode.className = 'badge badge-accent'; }
      else if (mode === 'offline') { el.agentMode.textContent = '模式 offline'; el.agentMode.className = 'badge badge-na'; }
      else if (mode) { el.agentMode.textContent = '模式 ' + mode; el.agentMode.className = 'badge badge-mid'; }
      else { el.agentMode.textContent = '模式 —'; el.agentMode.className = 'badge badge-na'; }
      el.agentMode.title = '';
    }
    if (el.agentModel) {
      el.agentModel.textContent = (d && d.model) ? ('模型 ' + String(d.model)) : '模型 —';
      el.agentModel.className = 'badge ' + ((d && d.model) ? 'badge-accent' : 'badge-na');
      el.agentModel.title = (d && d.model) ? String(d.model) : '';
    }
    if (el.agentStop) {
      var reason = (d && d.stop_reason !== undefined && d.stop_reason !== null) ? String(d.stop_reason) : '';
      el.agentStop.textContent = 'stop_reason ' + (reason || '—');
      el.agentStop.className = agentStopClass(reason);
      el.agentStop.title = agentStopText(reason);
    }
    if (el.agentHeadline) {
      var headline = (f && f.headline) ? String(f.headline) : '';
      el.agentHeadline.textContent = headline ? headline : (d ? '（final 缺失或 headline 为空）' : '—');
      el.agentHeadline.className = 'headline' + (headline ? '' : ' is-empty');
    }
    var c = (f && isNum(f.confidence)) ? f.confidence : null;
    if (el.agentConfNum) { el.agentConfNum.textContent = c === null ? '—' : c.toFixed(2); }
    if (el.agentConfFill) {
      var v = c === null ? 0 : c;
      if (v < 0) { v = 0; }
      if (v > 1) { v = 1; }
      el.agentConfFill.style.width = (v * 100).toFixed(1) + '%';
      el.agentConfFill.className = 'bar-fill' + ((c !== null && v < 0.4) ? ' is-neg' : '');
    }
  }

  function agentCallItem(c) {
    var call = (c && typeof c === 'object') ? c : {};
    var fail = (call.ok === false);
    var unknown = (call.ok !== true && call.ok !== false);
    var li = mk('li', 'agent-call' + (fail ? ' is-fail' : (unknown ? ' is-unknown' : '')));
    var head = mk('div', 'agent-call-head');
    head.appendChild(mk('span', 'agent-tool', call.name ? String(call.name) : '(未命名工具)'));
    if (fail) { head.appendChild(mk('span', 'badge badge-lo', 'ok=false')); }
    else if (call.ok === true) { head.appendChild(mk('span', 'badge badge-hi', 'ok=true')); }
    else { head.appendChild(mk('span', 'badge badge-na', 'ok —')); }
    if (isNum(call.duration_ms)) { head.appendChild(mk('span', 'agent-dur mono', fmtInt(call.duration_ms) + ' ms')); }
    li.appendChild(head);
    li.appendChild(mk('div', 'agent-args mono', 'args  ' + agentArgsText(call.args)));
    var summary = (typeof call.summary === 'string' && call.summary) ? call.summary : '';
    li.appendChild(mk('div', 'agent-summary' + (fail ? ' is-bad' : ''),
      summary ? summary : (fail ? '失败但未给出 summary（原因）' : '（summary 为空）')));
    var extra = '';
    if (typeof call.error === 'string' && call.error) { extra = call.error; }
    else if (typeof call.reason === 'string' && call.reason) { extra = call.reason; }
    if (extra && extra !== summary) { li.appendChild(mk('div', 'agent-reason is-bad', '原因：' + extra)); }
    if (call.evidence && typeof call.evidence === 'object') { li.appendChild(evidenceBlock('evidence', call.evidence)); }
    return li;
  }

  function renderAgentRounds() {
    var host = el.agentRounds;
    if (!host) { return; }
    clear(host);
    var d = agentData();
    if (el.agentRoundNote) {
      el.agentRoundNote.textContent = (d && d.mode === 'offline') ? '离线确定性计划（没有可用 LLM）' : '';
      show(el.agentRoundNote, !!(d && d.mode === 'offline'));
    }
    if (!d) {
      setText('agentRoundCount', '—');
      host.appendChild(mk('div', 'empty', agentEmptyText()));
      return;
    }
    var rounds = Array.isArray(d.rounds) ? d.rounds : [];
    setText('agentRoundCount', fmtInt(rounds.length) + ' 轮');
    if (!rounds.length) {
      host.appendChild(mk('div', 'empty', '响应里没有 rounds（rounds 缺失或为空）：本次运行没有发生任何轮次'));
      return;
    }
    for (var i = 0; i < rounds.length; i++) {
      var r = (rounds[i] && typeof rounds[i] === 'object') ? rounds[i] : {};
      var idx = isNum(r.index) ? r.index : (i + 1);
      var card = mk('div', 'agent-round');
      var head = mk('div', 'agent-round-head');
      head.appendChild(mk('span', 'agent-round-no', '第 ' + fmtInt(idx) + ' 轮'));
      var calls = Array.isArray(r.tool_calls) ? r.tool_calls : [];
      var fails = 0;
      for (var j = 0; j < calls.length; j++) {
        if (calls[j] && calls[j].ok === false) { fails++; }
      }
      head.appendChild(mk('span', 'chip', fmtInt(calls.length) + ' 次调用'));
      if (fails) { head.appendChild(mk('span', 'chip chip-err', fmtInt(fails) + ' 次失败')); }
      card.appendChild(head);
      var txt = (typeof r.assistant_text === 'string' && r.assistant_text) ? r.assistant_text : '';
      card.appendChild(mk('div', 'agent-round-text' + (txt ? '' : ' is-empty'), txt ? txt : '（本轮 assistant_text 为空）'));
      if (!calls.length) {
        card.appendChild(mk('div', 'empty', '本轮没有工具调用'));
      } else {
        var ul = mk('ul', 'agent-calls');
        for (j = 0; j < calls.length; j++) { ul.appendChild(agentCallItem(calls[j])); }
        card.appendChild(ul);
      }
      host.appendChild(card);
    }
  }

  function renderAgentSections() {
    var host = el.agentSections;
    if (!host) { return; }
    clear(host);
    var d = agentData();
    var f = (d && d.final && typeof d.final === 'object') ? d.final : null;
    if (!d) { host.appendChild(mk('div', 'empty', agentEmptyText())); return; }
    if (!f) { host.appendChild(mk('div', 'empty', '响应没有 final（final 字段缺失）：没有最终结论，请看轮次时间线')); return; }
    if (typeof f.text === 'string' && f.text) {
      host.appendChild(mk('div', 'report-sec-text', String(f.text)));
    }
    var sections = Array.isArray(f.sections) ? f.sections : [];
    if (!sections.length) { host.appendChild(mk('div', 'empty', 'final.sections 为空（sections 缺失或没有内容）')); return; }
    for (var i = 0; i < sections.length; i++) {
      var s = (sections[i] && typeof sections[i] === 'object') ? sections[i] : {};
      var card = mk('section', 'report-sec');
      var head = mk('div', 'report-sec-head');
      head.appendChild(mk('h3', 'report-sec-title', s.title ? String(s.title) : '(无标题)'));
      if (s.id) { head.appendChild(mk('span', 'report-sec-id', String(s.id))); }
      card.appendChild(head);
      if (s.text) { card.appendChild(mk('div', 'report-sec-text', String(s.text))); }
      card.appendChild(narrateEvidenceTable(s.evidence));
      host.appendChild(card);
    }
  }

  function agentList(host, countEl, list, kind, emptyMsg) {
    if (!host) { return; }
    clear(host);
    var d = agentData();
    var f = (d && d.final && typeof d.final === 'object') ? d.final : null;
    var arr = Array.isArray(list) ? list : [];
    if (countEl) { countEl.textContent = f ? (fmtInt(arr.length) + ' 条') : '—'; }
    if (!d) { host.appendChild(mk('div', 'empty', agentEmptyText())); return; }
    if (!f) { host.appendChild(mk('div', 'empty', 'final 缺失：没有不确定项 / 下一步动作')); return; }
    if (!arr.length) { host.appendChild(mk('div', 'empty', emptyMsg)); return; }
    var ul = mk('ul', 'rep-list is-' + kind);
    for (var i = 0; i < arr.length; i++) { ul.appendChild(mk('li', null, String(arr[i]))); }
    host.appendChild(ul);
  }

  function renderAgentLists() {
    var d = agentData();
    var f = (d && d.final && typeof d.final === 'object') ? d.final : null;
    agentList(el.agentUncertainties, el.agentUncCount, f && f.uncertainties, 'unc', 'uncertainties 为空');
    agentList(el.agentActions, el.agentActCount, f && f.next_actions, 'act', 'next_actions 为空');
  }

  function renderAgentSummary() {
    var d = agentData();
    var trace = (d && d.trace_summary && typeof d.trace_summary === 'object') ? d.trace_summary : null;
    if (el.agentElapsed) { el.agentElapsed.textContent = d ? fmtElapsed(state.agent.elapsedMs) : '—'; }
    if (el.agentTrace) {
      clear(el.agentTrace);
      if (!d) { el.agentTrace.appendChild(mk('div', 'empty', agentEmptyText())); }
      else if (!trace) { el.agentTrace.appendChild(mk('div', 'empty', '响应没有 trace_summary（运行摘要缺失）')); }
      else {
        var rounds = isNum(trace.rounds) ? trace.rounds : (Array.isArray(d.rounds) ? d.rounds.length : null);
        var calls = isNum(trace.tool_calls) ? trace.tool_calls : agentCountCalls(d);
        llmKvRow(el.agentTrace, 'rounds', rounds === null ? '—' : fmtInt(rounds));
        llmKvRow(el.agentTrace, 'tool_calls', calls === null ? '—' : fmtInt(calls));
        var used = Array.isArray(trace.tools_used) ? trace.tools_used : null;
        var usedTxt = used ? (used.length ? used.join(', ') : '（空）') : '—';
        var usedRow = llmKvRow(el.agentTrace, 'tools_used', usedTxt);
        if (used && used.length) { usedRow.title = used.join(', '); }
        llmKvRow(el.agentTrace, 'elapsed_ms', isNum(trace.elapsed_ms) ? (fmtInt(trace.elapsed_ms) + ' ms') : '—');
        var b = (trace.budget && typeof trace.budget === 'object') ? trace.budget : agentBudget();
        if (b) {
          var bits = [];
          if (isNum(b.max_rounds)) { bits.push('rounds ≤ ' + fmtInt(b.max_rounds)); }
          if (isNum(b.max_tool_calls)) { bits.push('calls ≤ ' + fmtInt(b.max_tool_calls)); }
          if (isNum(b.timeout_s)) { bits.push('timeout ≤ ' + fmtInt(b.timeout_s) + ' s'); }
          llmKvRow(el.agentTrace, 'budget', bits.length ? bits.join(' / ') : '—');
        } else {
          llmKvRow(el.agentTrace, 'budget', '—');
        }
      }
    }
    if (el.agentStopText) {
      var reason = (d && d.stop_reason !== undefined && d.stop_reason !== null) ? String(d.stop_reason) : '';
      el.agentStopText.textContent = d ? agentStopText(reason) : agentEmptyText();
      el.agentStopText.className = 'agent-stop-text' + (d ? agentStopSuffix(reason) : '');
    }
  }

  function renderAgentGrounding() {
    var d = agentData();
    var g = (d && d.grounding && typeof d.grounding === 'object') ? d.grounding : null;
    var grounded = g ? (g.grounded === true) : null;
    if (el.agentGroundingChip) {
      var cls = 'chip';
      var txt = '—';
      if (!d) { txt = '—'; }
      else if (!g) { txt = 'grounding 缺失'; cls += ' chip-warn'; }
      else if (grounded) { txt = 'grounded=true'; cls += ' chip-ok'; }
      else { txt = 'grounded=false'; cls += ' chip-err'; }
      el.agentGroundingChip.textContent = txt;
      el.agentGroundingChip.className = cls;
    }
    if (el.agentGrounding) {
      clear(el.agentGrounding);
      if (!d) { el.agentGrounding.appendChild(mk('div', 'empty', agentEmptyText())); }
      else if (!g) { el.agentGrounding.appendChild(mk('div', 'empty', '响应没有 grounding：本次结论未做数字锚定校验')); }
      else {
        llmKvRow(el.agentGrounding, 'grounded', grounded ? 'true' : 'false', grounded ? 'is-ok' : 'is-err');
        llmKvRow(el.agentGrounding, 'checked_numbers', isNum(g.checked_numbers) ? fmtInt(g.checked_numbers) : '—');
        var un = Array.isArray(g.unsupported) ? g.unsupported : [];
        llmKvRow(el.agentGrounding, 'unsupported', fmtInt(un.length) + ' 项', un.length ? 'is-err' : 'is-ok');
      }
    }
    if (el.agentUnsupported) {
      clear(el.agentUnsupported);
      var showUn = false;
      if (d && g && grounded === false) {
        var list = Array.isArray(g.unsupported) ? g.unsupported : [];
        el.agentUnsupported.appendChild(mk('span', 'agent-unsupported-k',
          'grounded=false：以下数字在证据里找不到出处（LLM 可能编造），请勿直接采信'));
        var ul = mk('ul');
        if (!list.length) { ul.appendChild(mk('li', 'is-empty', 'unsupported 为空 —— 后端未给出具体数字')); }
        else { for (var i = 0; i < list.length; i++) { ul.appendChild(mk('li', null, String(list[i]))); } }
        el.agentUnsupported.appendChild(ul);
        showUn = true;
      } else if (d && !g) {
        el.agentUnsupported.appendChild(mk('span', 'agent-unsupported-k', 'grounding 缺失：无法判定数字是否被证据支持'));
        showUn = true;
      }
      show(el.agentUnsupported, showUn);
    }
  }

  function renderAgentPrompt() {
    if (el.agentPrompt) {
      var d = agentData();
      var p = (d && typeof d.prompt_preview === 'string') ? d.prompt_preview : '';
      el.agentPrompt.textContent = p ? p : (d ? '（prompt_preview 为空）' : agentEmptyText());
    }
    if (el.agentPromptFold) { show(el.agentPromptFold, true); }
  }

  function renderAgentWarnings() {
    var host = el.agentWarnings;
    if (!host) { return; }
    clear(host);
    var d = agentData();
    if (!d) {
      setText('agentWarnCount', '—');
      host.appendChild(mk('div', 'empty', agentEmptyText()));
      return;
    }
    var list = Array.isArray(d.warnings) ? d.warnings : [];
    setText('agentWarnCount', fmtInt(list.length) + ' 条');
    if (!list.length) { host.appendChild(mk('div', 'empty', '无告警')); return; }
    var ul = mk('ul', 'warnlist');
    for (var i = 0; i < list.length; i++) { ul.appendChild(mk('li', null, String(list[i]))); }
    host.appendChild(ul);
  }

  function renderAgent() {
    renderAgentBudget();
    renderAgentStatus();
    renderAgentHeader();
    renderAgentRounds();
    renderAgentSections();
    renderAgentLists();
    renderAgentSummary();
    renderAgentGrounding();
    renderAgentPrompt();
    renderAgentWarnings();
  }

  function syncAgentControls() {
    if (el.ctlAgentProvider) { el.ctlAgentProvider.value = state.ui.agentProvider || 'auto'; }
    if (el.ctlAgentRounds) {
      var v = agentMaxRounds();
      /* 用户手动改过就不覆盖（tools 迟到时不能把用户输入冲掉） */
      if (!state.ui.agentRoundsManual || !el.ctlAgentRounds.value) { el.ctlAgentRounds.value = String(v); }
      el.ctlAgentRounds.max = '12';
    }
  }

  /* ================= 标签页 ================= */
  function activateTab(tab, opts) {
    opts = opts || {};
    if (TABS.indexOf(tab) < 0) { tab = 'spectrum'; }
    state.activeTab = tab;
    var i;
    for (i = 0; i < el.tabButtons.length; i++) {
      var b = el.tabButtons[i];
      var on = b.getAttribute('data-tab') === tab;
      if (on) { b.className = 'tab is-active'; b.setAttribute('aria-selected', 'true'); b.removeAttribute('tabindex'); }
      else { b.className = 'tab'; b.setAttribute('aria-selected', 'false'); b.setAttribute('tabindex', '-1'); }
    }
    for (i = 0; i < el.tabPanels.length; i++) {
      var p = el.tabPanels[i];
      show(p, p.getAttribute('data-panel') === tab);
    }
    if (!opts.skipUrl) { syncUrl(false); }
    renderStatus();
    if (!opts.skipData) { ensureTabData(tab); }
    redrawVisible();
  }

  function redrawVisible() {
    if (state.activeTab === 'spectrum') {
      drawSpectrum();
      drawWaterfall();
    } else if (state.activeTab === 'features') {
      drawConstellation();
    } else if (state.activeTab === 'demod') {
      drawDemodConstellation();
      drawSyncTraces();
    } else if (state.activeTab === 'fec') {
      drawFecCurves();
    } else if (state.activeTab === 'frame') {
      drawFrameCorr();
    } else if (state.activeTab === 'ssca') {
      drawSscaMap();
      drawSscaAlpha();
      drawSscaFreq();
    }
  }

  /* ================= 事件绑定 ================= */
  function onResize() {
    if (state.resizeRaf) { return; }
    state.resizeRaf = requestAnimationFrame(function () {
      state.resizeRaf = 0;
      redrawVisible();
    });
  }

  function bindEvents() {
    var loadBtn = el.pathLoad;
    if (loadBtn) {
      loadBtn.addEventListener('click', function () {
        var v = el.pathInput && el.pathInput.value ? el.pathInput.value.trim() : '';
        if (!v) {
          recordError('客户端', 0, '路径为空：请输入 .sigmf-meta / .npz / .wav / .raw 路径', '');
          return;
        }
        loadPath(v, { push: true });
      });
    }
    if (el.pathInput) {
      el.pathInput.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') { ev.preventDefault(); if (loadBtn) { loadBtn.click(); } }
      });
    }
    if (el.dirLoad) {
      el.dirLoad.addEventListener('click', function () {
        var v = el.dirInput && el.dirInput.value ? el.dirInput.value.trim() : '';
        state.dir = v || 'data/demo';
        if (el.dirInput) { el.dirInput.value = state.dir; }
        loadCases();
      });
    }
    if (el.dirInput) {
      el.dirInput.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') { ev.preventDefault(); if (el.dirLoad) { el.dirLoad.click(); } }
      });
    }
    if (el.caseFilter) {
      el.caseFilter.addEventListener('input', function () {
        state.filter = el.caseFilter.value || '';
        renderCases();
      });
    }
    if (el.errorClear) { el.errorClear.addEventListener('click', clearErrors); }
    if (el.errorRetry) { el.errorRetry.addEventListener('click', retryAll); }

    var i;
    for (i = 0; i < el.tabButtons.length; i++) {
      (function (btn) {
        btn.addEventListener('click', function () { activateTab(btn.getAttribute('data-tab')); });
        btn.addEventListener('keydown', function (ev) {
          var idx = TABS.indexOf(btn.getAttribute('data-tab'));
          var next = -1;
          if (ev.key === 'ArrowRight') { next = (idx + 1) % TABS.length; }
          else if (ev.key === 'ArrowLeft') { next = (idx - 1 + TABS.length) % TABS.length; }
          else if (ev.key === 'Home') { next = 0; }
          else if (ev.key === 'End') { next = TABS.length - 1; }
          if (next < 0) { return; }
          ev.preventDefault();
          var target = null;
          for (var j = 0; j < el.tabButtons.length; j++) {
            if (el.tabButtons[j].getAttribute('data-tab') === TABS[next]) { target = el.tabButtons[j]; }
          }
          if (target) { activateTab(TABS[next]); target.focus(); }
        });
      }(el.tabButtons[i]));
    }

    var np = el.ctlNperseg;
    if (np) {
      np.addEventListener('change', function () {
        state.ui.nperseg = parseInt(np.value, 10) || 1024;
        renderSpectrumLegend();
        loadSpectrum();
        loadWaterfall();
      });
    }
    var nf = el.ctlNfft;
    if (nf) {
      nf.addEventListener('change', function () {
        state.ui.nfft = nf.value ? (parseInt(nf.value, 10) || null) : null;
        loadSpectrum();
      });
    }
    var pt = el.ctlPoints;
    if (pt) {
      pt.addEventListener('change', function () {
        state.ui.points = parseInt(pt.value, 10) || 1200;
        loadSpectrum();
      });
    }
    var rl = el.ctlReload;
    if (rl) {
      rl.addEventListener('click', function () {
        loadSpectrum();
        loadWaterfall();
      });
    }
    var res = el.ctlResidual;
    if (res) {
      res.addEventListener('change', function () {
        state.ui.residual = !!res.checked;
        renderSpectrumLegend();
        drawSpectrum();
      });
    }

    /* 解调页控件：modulation / max_symbols / trace_points 任一变化都重新请求 /api/demod */
    var dmod = el.ctlDemodMod;
    if (dmod) {
      dmod.addEventListener('change', function () {
        state.ui.demodMod = dmod.value || '';
        syncUrl(false);
        loadDemod();
      });
    }
    var dms = el.ctlDemodMaxSymbols;
    if (dms) {
      dms.addEventListener('change', function () {
        state.ui.demodMaxSymbols = parseInt(dms.value, 10) || 2000;
        loadDemod();
      });
    }
    var dtp = el.ctlDemodTracePoints;
    if (dtp) {
      dtp.addEventListener('change', function () {
        state.ui.demodTracePoints = parseInt(dtp.value, 10) || 400;
        loadDemod();
      });
    }
    var dre = el.ctlDemodRegions;
    if (dre) {
      dre.addEventListener('change', function () {
        state.ui.demodRegions = !!dre.checked;
        renderDemodLegends();
        drawDemodConstellation();
      });
    }
    var drl = el.ctlDemodReload;
    if (drl) {
      drl.addEventListener('click', function () { loadDemod(); });
    }

    /* 谱相关页：nfft / max_points 变化重新计算，「重新计算」按钮显式重试失败态 */
    var snf = el.ctlSscaNfft;
    if (snf) {
      snf.addEventListener('change', function () {
        state.ui.sscaNfft = parseInt(snf.value, 10) || 256;
        loadSsca();
      });
    }
    var smp = el.ctlSscaPoints;
    if (smp) {
      smp.addEventListener('change', function () {
        state.ui.sscaMaxPoints = parseInt(smp.value, 10) || 160;
        loadSsca();
      });
    }
    var srl = el.ctlSscaReload;
    if (srl) { srl.addEventListener('click', function () { loadSsca(); }); }
    if (el.sscaCanvas) {
      el.sscaCanvas.addEventListener('mousemove', sscaOnMove);
      el.sscaCanvas.addEventListener('mouseleave', sscaHoverClear);
    }

    /* 报告页：provider 切换 / 「重新生成」都显式重发（失败态也靠它重试） */
    var npr = el.ctlNarrateProvider;
    if (npr) {
      npr.addEventListener('change', function () {
        var v = npr.value;
        state.ui.narrateProvider = (v === 'offline' || v === 'llm') ? v : 'auto';
        loadNarrate();
      });
    }
    if (el.ctlNarrateReload) { el.ctlNarrateReload.addEventListener('click', function () { loadNarrate(); }); }
    if (el.narrateGotoSettings) { el.narrateGotoSettings.addEventListener('click', function () { activateTab('settings'); }); }

    /* 设置页（LLM 配置）：只保留"用户动作 -> 请求/状态"的连线，渲染全部走 renderLlmSettings */
    if (el.llmRefresh) { el.llmRefresh.addEventListener('click', function () { loadLlmConfig(); }); }
    if (el.llmSave) { el.llmSave.addEventListener('click', function () { saveLlmConfig(); }); }
    if (el.llmTest) { el.llmTest.addEventListener('click', function () { testLlm(); }); }
    if (el.llmModelsPull) { el.llmModelsPull.addEventListener('click', function () { loadLlmModels(); }); }
    if (el.llmKeyEye) { el.llmKeyEye.addEventListener('click', function () { setLlmKeyVisible(!state.llm.showKey); }); }
    if (el.llmKeyClear) { el.llmKeyClear.addEventListener('click', function () { clearLlmKey(); }); }
    if (el.llmApiKey) {
      /* 密钥只存在内存草稿 + password 输入框；input 事件不写任何页面文本、不进任何日志 */
      el.llmApiKey.addEventListener('input', function () {
        state.llm.apiKeyDraft = el.llmApiKey.value ? String(el.llmApiKey.value) : '';
      });
      /* 阻止浏览器把输入框内容当表单提交（本页无 form，纯防御） */
      el.llmApiKey.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') { ev.preventDefault(); saveLlmConfig(); }
      });
    }
    if (el.llmModelSelect) {
      el.llmModelSelect.addEventListener('change', function () {
        var v = el.llmModelSelect.value;
        if (v && el.llmModel) { el.llmModel.value = String(v); }
        renderLlmPresets();
      });
    }
    /* base_url / model 手输时实时刷新预设高亮（纯本地判定，不发请求） */
    if (el.llmBaseUrl) { el.llmBaseUrl.addEventListener('input', function () { renderLlmPresets(); }); }
    if (el.llmModel) { el.llmModel.addEventListener('input', function () { renderLlmPresets(); }); }
    if (el.llmTemperatureRange) {
      el.llmTemperatureRange.addEventListener('input', function () {
        if (el.llmTemperature) { el.llmTemperature.value = String(el.llmTemperatureRange.value); }
      });
    }
    if (el.llmTemperature) {
      el.llmTemperature.addEventListener('input', function () {
        var v = Number(el.llmTemperature.value);
        if (isFinite(v) && el.llmTemperatureRange) { el.llmTemperatureRange.value = String(v); }
      });
    }

    /* 代理页：只有用户点「运行代理」才 POST；进入页面 / 切标签都不会自动运行 */
    if (el.agentRun) { el.agentRun.addEventListener('click', runAgent); }
    if (el.agentToolsReload) { el.agentToolsReload.addEventListener('click', function () { loadAgentTools(); }); }
    if (el.ctlAgentProvider) {
      el.ctlAgentProvider.addEventListener('change', function () { state.ui.agentProvider = agentProviderValue(); });
    }
    if (el.ctlAgentRounds) {
      el.ctlAgentRounds.addEventListener('change', function () {
        state.ui.agentRoundsManual = true;
        el.ctlAgentRounds.value = String(agentRoundsValue());
      });
    }
    if (el.agentGoal) {
      el.agentGoal.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') { ev.preventDefault(); runAgent(); }
      });
    }

    /* 访问令牌：保存 / 清除 / 刷新都只动本地状态；令牌不发给后端保存 */
    if (el.authTokenSave) { el.authTokenSave.addEventListener('click', saveAuthToken); }
    if (el.authTokenClear) { el.authTokenClear.addEventListener('click', clearAuthToken); }
    if (el.authRefresh) { el.authRefresh.addEventListener('click', function () { loadAuthStatus(); }); }
    if (el.authToken) {
      el.authToken.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') { ev.preventDefault(); saveAuthToken(); }
      });
    }

    /* 译码页：重新加载 + 纵轴指标切换；图例开关在 renderFecLegend() 里逐个绑定 */
    var frl = el.ctlFecReload;
    if (frl) { frl.addEventListener('click', function () { loadFec(); }); }
    if (el.ctlFrameReload) { el.ctlFrameReload.addEventListener('click', function () { loadFrame(); }); }
    if (el.ctlFrameMod) {
      el.ctlFrameMod.addEventListener('change', function () {
        state.ui.frameMod = el.ctlFrameMod.value || '';
        loadFrame();
      });
    }
    if (el.frameTable) {
      el.frameTable.addEventListener('click', function (event) {
        var node = event.target;
        while (node && node !== el.frameTable) {
          if (node.getAttribute && node.getAttribute('data-frame-index') !== null) {
            state.ui.frameSel = parseInt(node.getAttribute('data-frame-index'), 10) || 0;
            renderFrameTable();
            renderFramePayload();
            return;
          }
          node = node.parentNode;
        }
      });
    }
    if (el.fecMetricBer) { el.fecMetricBer.addEventListener('click', function () { setFecMetric('ber'); }); }
    if (el.fecMetricBler) { el.fecMetricBler.addEventListener('click', function () { setFecMetric('bler'); }); }
    if (el.fecCanvas) {
      el.fecCanvas.addEventListener('mousemove', fecOnMove);
      el.fecCanvas.addEventListener('mouseleave', fecOnLeave);
    }
    var segs = document.querySelectorAll('.seg-btn');
    for (i = 0; i < segs.length; i++) {
      (function (b) {
        /* 只有频谱页的幅度刻度按钮带 data-scale；译码页的 BER/BLER 分段按钮不归这里管 */
        if (!b.getAttribute('data-scale')) { return; }
        b.addEventListener('click', function () {
          state.ui.scale = b.getAttribute('data-scale') === 'linear' ? 'linear' : 'db';
          for (var j = 0; j < segs.length; j++) {
            var on = segs[j].getAttribute('data-scale') === state.ui.scale;
            segs[j].className = on ? 'seg-btn is-active' : 'seg-btn';
            segs[j].setAttribute('aria-pressed', on ? 'true' : 'false');
          }
          drawSpectrum();
          renderSpectrumLegend();
        });
      }(segs[i]));
    }

    window.addEventListener('resize', onResize);
    window.addEventListener('popstate', function () {
      var prev = state.path;
      readUrl();
      if (state.path && state.path !== prev) {
        loadPath(state.path, { push: false });
        activateTab(state.activeTab, { skipUrl: true });
      } else {
        activateTab(state.activeTab, { skipUrl: true });
      }
    });
    if (typeof ResizeObserver !== 'undefined') {
      var ro = new ResizeObserver(onResize);
      if (el.workspace) { ro.observe(el.workspace); }
      if (el.sidebar) { ro.observe(el.sidebar); }
    }
  }

  /* ================= 启动 ================= */
  function cacheEls() {
    el.errorBar = $('errorbar');
    el.errorList = $('errorlist');
    el.errorCount = $('error-count');
    el.errorClear = $('error-clear');
    el.errorRetry = $('error-retry');
    el.healthEstimator = $('health-estimator');
    el.healthClassifier = $('health-classifier');
    el.modeChip = $('mode-chip');
    el.sidebar = $('sidebar');
    el.workspace = $('workspace');
    el.pathInput = $('path-input');
    el.pathLoad = $('path-load');
    el.dirInput = $('dir-input');
    el.dirLoad = $('dir-load');
    el.caseFilter = $('case-filter');
    el.caseList = $('case-list');
    el.caseEmpty = $('case-empty');
    el.caseCount = $('case-count');
    el.alertStrip = $('alertstrip');
    el.alertList = $('alertlist');
    el.ovPath = $('ov-path');
    el.ovFs = $('ov-fs');
    el.ovN = $('ov-n');
    el.ovDur = $('ov-dur');
    el.ovPeak = $('ov-peak');
    el.ovPeakSnr = $('ov-peak-snr');
    el.ovBw = $('ov-bw');
    el.ovFloor = $('ov-floor');
    el.spectrumCanvas = $('spectrum-canvas');
    el.spectrumEmpty = $('spectrum-empty');
    el.spectrumLegend = $('spectrum-legend');
    el.waterfallCanvas = $('waterfall-canvas');
    el.waterfallEmpty = $('waterfall-empty');
    el.waterfallMeta = $('waterfall-meta');
    el.constellationCanvas = $('constellation-canvas');
    el.constellationEmpty = $('constellation-empty');
    el.constellationMeta = $('constellation-meta');
    el.estCards = $('est-cards');
    el.compareBody = $('compare-body');
    el.compareMod = $('compare-mod');
    el.analyzeMeta = $('analyze-meta');
    el.featureSummary = $('feature-summary');
    el.featureGroups = $('feature-groups');
    el.verdictMod = $('verdict-mod');
    el.verdictSub = $('verdict-sub');
    el.verdictGate = $('verdict-gate');
    el.verdictConf = $('verdict-conf');
    el.verdictOod = $('verdict-ood');
    el.verdictCal = $('verdict-cal');
    el.probBars = $('prob-bars');
    el.probCount = $('prob-count');
    el.gateBody = $('gate-body');
    el.physicalBody = $('physical-body');
    el.physicalState = $('physical-state');
    el.lockTiming = $('lock-timing');
    el.lockCarrier = $('lock-carrier');
    el.lockMod = $('lock-mod');
    el.lockSrc = $('lock-src');
    el.lockTimingMetric = $('lock-timing-metric');
    el.lockCarrierMetric = $('lock-carrier-metric');
    el.lockSnr = $('lock-snr');
    el.lockSyms = $('lock-syms');
    el.lockEvm = $('lock-evm');
    el.demodConstellationCanvas = $('demod-constellation-canvas');
    el.demodConstellationEmpty = $('demod-constellation-empty');
    el.demodConstLegend = $('demod-const-legend');
    el.demodConstMeta = $('demod-const-meta');
    el.demodSyncCanvas = $('demod-sync-canvas');
    el.demodSyncEmpty = $('demod-sync-empty');
    el.demodSyncLegend = $('demod-sync-legend');
    el.demodSyncMeta = $('demod-sync-meta');
    el.bitsBody = $('bits-body');
    el.bitsCount = $('bits-count');
    el.berBody = $('ber-body');
    el.berState = $('ber-state');
    el.demodWarnings = $('demod-warnings');
    el.demodWarnCount = $('demod-warn-count');
    el.ctlDemodMod = $('ctl-demod-mod');
    el.ctlDemodMaxSymbols = $('ctl-demod-max-symbols');
    el.ctlDemodTracePoints = $('ctl-demod-trace-points');
    el.ctlDemodRegions = $('ctl-demod-regions');
    el.ctlDemodReload = $('ctl-demod-reload');
    el.fsLock = $('fs-lock');
    el.fsPreamble = $('fs-preamble');
    el.fsMod = $('fs-mod');
    el.fsSrc = $('fs-src');
    el.fsPeak = $('fs-peak');
    el.fsPsl = $('fs-psl');
    el.fsSymbol = $('fs-symbol');
    el.fsSample = $('fs-sample');
    el.fsPreambleBits = $('fs-preamble-bits');
    el.rotNeedle = $('rot-needle');
    el.rotDeg = $('rot-deg');
    el.rotQuad = $('rot-quad');
    el.rotNote = $('rot-note');
    el.frameCards = $('frame-cards');
    el.frameCorrCanvas = $('frame-corr-canvas');
    el.frameCorrEmpty = $('frame-corr-empty');
    el.frameCorrLegend = $('frame-corr-legend');
    el.frameCorrMeta = $('frame-corr-meta');
    el.frameCount = $('frame-count');
    el.frameSelChip = $('frame-sel-chip');
    el.frameTable = $('frame-table');
    el.payloadMeta = $('payload-meta');
    el.payloadBody = $('payload-body');
    el.frameWarnCount = $('frame-warn-count');
    el.frameWarnings = $('frame-warnings');
    el.ctlFrameMod = $('ctl-frame-mod');
    el.ctlFrameReload = $('ctl-frame-reload');
    el.fecCards = $('fec-cards');
    el.fecLegend = $('fec-legend');
    el.fecCanvas = $('fec-canvas');
    el.fecEmpty = $('fec-empty');
    el.fecTip = $('fec-tip');
    el.fecPlotBody = $('fec-plot-body');
    el.fecPlotTitle = $('fec-plot-title');
    el.fecMeta = $('fec-meta');
    el.fecCatalogue = $('fec-catalogue');
    el.fecCatCount = $('fec-cat-count');
    el.fecChecks = $('fec-checks');
    el.fecCheckCount = $('fec-check-count');
    el.fecNotes = $('fec-notes');
    el.fecNoteCount = $('fec-note-count');
    el.ctlFecReload = $('ctl-fec-reload');
    el.fecMetricBer = $('fec-metric-ber');
    el.fecMetricBler = $('fec-metric-bler');
    el.sscaCanvas = $('ssca-canvas');
    el.sscaEmpty = $('ssca-empty');
    el.sscaTip = $('ssca-tip');
    el.sscaLegend = $('ssca-legend');
    el.sscaMapMeta = $('ssca-map-meta');
    el.sscaAlphaCanvas = $('ssca-alpha-canvas');
    el.sscaAlphaEmpty = $('ssca-alpha-empty');
    el.sscaAlphaMeta = $('ssca-alpha-meta');
    el.sscaFreqCanvas = $('ssca-freq-canvas');
    el.sscaFreqEmpty = $('ssca-freq-empty');
    el.sscaFreqMeta = $('ssca-freq-meta');
    el.sscaFeatures = $('ssca-features');
    el.sscaPeaks = $('ssca-peaks');
    el.sscaPeakCount = $('ssca-peak-count');
    el.sscaWarnings = $('ssca-warnings');
    el.sscaWarnCount = $('ssca-warn-count');
    el.sscaMeta = $('ssca-meta');
    el.ctlSscaNfft = $('ctl-ssca-nfft');
    el.ctlSscaPoints = $('ctl-ssca-points');
    el.ctlSscaReload = $('ctl-ssca-reload');
    el.statusPath = $('status-path');
    el.statusTab = $('status-tab');
    el.statusUpdated = $('status-updated');
    el.statusSource = $('status-source');
    el.ctlNperseg = $('ctl-nperseg');
    el.ctlNfft = $('ctl-nfft');
    el.ctlPoints = $('ctl-points');
    el.ctlResidual = $('ctl-residual');
    el.ctlReload = $('ctl-reload');
    el.frameBlind = $('frame-blind');
    el.fbHeaderCrc = $('fb-header-crc');
    el.fbLength = $('fb-length');
    el.fbHypotheses = $('fb-hypotheses');
    el.fbPayloadLen = $('fb-payload-len');
    el.fbFec = $('fb-fec');
    el.fbInterleave = $('fb-interleave');
    el.narrateMeta = $('narrate-meta');
    el.narrateMode = $('narrate-mode');
    el.narrateModel = $('narrate-model');
    el.narrateTime = $('narrate-time');
    el.narrateConfNum = $('narrate-conf-num');
    el.narrateConfFill = $('narrate-conf-fill');
    el.narrateHeadline = $('narrate-headline');
    el.narrateEmpty = $('narrate-empty');
    el.narrateSections = $('narrate-sections');
    el.narrateUncertainties = $('narrate-uncertainties');
    el.narrateUncCount = $('narrate-unc-count');
    el.narrateActions = $('narrate-actions');
    el.narrateActCount = $('narrate-act-count');
    el.narratePromptFold = $('narrate-prompt-fold');
    el.narratePrompt = $('narrate-prompt');
    el.narrateWarnings = $('narrate-warnings');
    el.narrateWarnCount = $('narrate-warn-count');
    el.ctlNarrateProvider = $('ctl-narrate-provider');
    el.ctlNarrateReload = $('ctl-narrate-reload');
    el.narrateLlmHint = $('narrate-llm-hint');
    el.narrateGotoSettings = $('narrate-goto-settings');
    el.narrateLlmInfo = $('narrate-llm-info');
    /* 设置页（LLM 配置） */
    el.llmReady = $('llm-ready');
    el.llmMeta = $('llm-meta');
    el.llmRefresh = $('llm-refresh');
    el.llmReadyBanner = $('llm-ready-banner');
    el.llmReadyChip = $('llm-ready-chip');
    el.llmConfigPath = $('llm-config-path');
    el.llmEnabled = $('llm-enabled');
    el.llmSrcEnabled = $('llm-src-enabled');
    el.llmPresets = $('llm-presets');
    el.llmPresetNotes = $('llm-preset-notes');
    el.llmBaseUrl = $('llm-base-url');
    el.llmSrcBaseUrl = $('llm-src-base-url');
    el.llmApiKey = $('llm-api-key');
    el.llmKeyEye = $('llm-key-eye');
    el.llmSrcApiKey = $('llm-src-api-key');
    el.llmKeyState = $('llm-key-state');
    el.llmKeyClear = $('llm-key-clear');
    el.llmModel = $('llm-model');
    el.llmModelOptions = $('llm-model-options');
    el.llmModelSelect = $('llm-model-select');
    el.llmModelsPull = $('llm-models-pull');
    el.llmModelsNote = $('llm-models-note');
    el.llmSrcModel = $('llm-src-model');
    el.llmTemperature = $('llm-temperature');
    el.llmTemperatureRange = $('llm-temperature-range');
    el.llmSrcTemperature = $('llm-src-temperature');
    el.llmMaxTokens = $('llm-max-tokens');
    el.llmSrcMaxTokens = $('llm-src-max-tokens');
    el.llmTimeout = $('llm-timeout');
    el.llmSrcTimeout = $('llm-src-timeout-s');
    el.llmSave = $('llm-save');
    el.llmTest = $('llm-test');
    el.llmTestOverride = $('llm-test-override');
    el.llmSaveStatus = $('llm-save-status');
    el.llmSummary = $('llm-summary');
    el.llmEnv = $('llm-env');
    el.llmTestChip = $('llm-test-chip');
    el.llmTestResult = $('llm-test-result');
    /* 代理页（契约 §18） */
    el.agentGoal = $('agent-goal');
    el.ctlAgentProvider = $('ctl-agent-provider');
    el.ctlAgentRounds = $('ctl-agent-rounds');
    el.agentRun = $('agent-run');
    el.agentToolsReload = $('agent-tools-reload');
    el.agentBudget = $('agent-budget');
    el.agentStatus = $('agent-status');
    el.agentMode = $('agent-mode');
    el.agentModel = $('agent-model');
    el.agentStop = $('agent-stop');
    el.agentConfNum = $('agent-conf-num');
    el.agentConfFill = $('agent-conf-fill');
    el.agentHeadline = $('agent-headline');
    el.agentEmpty = $('agent-empty');
    el.agentRoundCount = $('agent-round-count');
    el.agentRoundNote = $('agent-round-note');
    el.agentRounds = $('agent-rounds');
    el.agentSections = $('agent-sections');
    el.agentUncCount = $('agent-unc-count');
    el.agentUncertainties = $('agent-uncertainties');
    el.agentActCount = $('agent-act-count');
    el.agentActions = $('agent-actions');
    el.agentElapsed = $('agent-elapsed');
    el.agentTrace = $('agent-trace');
    el.agentStopText = $('agent-stop-text');
    el.agentGroundingChip = $('agent-grounding-chip');
    el.agentGrounding = $('agent-grounding');
    el.agentUnsupported = $('agent-unsupported');
    el.agentPromptFold = $('agent-prompt-fold');
    el.agentPrompt = $('agent-prompt');
    el.agentWarnCount = $('agent-warn-count');
    el.agentWarnings = $('agent-warnings');
    /* 访问令牌（契约 §19） */
    el.authState = $('auth-state');
    el.authHint = $('auth-hint');
    el.authRefresh = $('auth-refresh');
    el.authToken = $('auth-token');
    el.authTokenSave = $('auth-token-save');
    el.authTokenClear = $('auth-token-clear');
    el.authTokenState = $('auth-token-state');
    el.tabButtons = [].slice.call(document.querySelectorAll('.tab'));
    el.tabPanels = [].slice.call(document.querySelectorAll('.panel-tab'));
  }

  function syncSscaControls() {
    if (el.ctlSscaNfft) { el.ctlSscaNfft.value = String(state.ui.sscaNfft); }
    if (el.ctlSscaPoints) { el.ctlSscaPoints.value = String(state.ui.sscaMaxPoints); }
  }

  function syncDemodControls() {
    if (el.ctlDemodMod) { el.ctlDemodMod.value = state.ui.demodMod || ''; }
    if (el.ctlDemodMaxSymbols) { el.ctlDemodMaxSymbols.value = String(state.ui.demodMaxSymbols); }
    if (el.ctlDemodTracePoints) { el.ctlDemodTracePoints.value = String(state.ui.demodTracePoints); }
    if (el.ctlDemodRegions) { el.ctlDemodRegions.checked = !!state.ui.demodRegions; }
  }

  function init() {
    cacheEls();
    readUrl();
    if (el.modeChip) {
      el.modeChip.textContent = state.mock ? 'mock 数据' : '后端';
      el.modeChip.className = 'meta-chip' + (state.mock ? ' is-warn' : '');
      el.modeChip.title = state.mock ? '?mock=1：从 /static/mock/*.json 读假数据' : '从 /api/* 读真实结果';
    }
    if (el.dirInput) { el.dirInput.value = state.dir; }
    if (el.pathInput) { el.pathInput.value = state.path || ''; }
    /* 令牌只在内存 + localStorage：这里读一次进内存，绝不回填到任何 DOM 节点 */
    state.auth.token = readStoredToken();
    state.auth.persistFailed = false;
    syncDemodControls();
    syncSscaControls();
    syncNarrateControls();
    syncAgentControls();
    syncFecMetricButtons();
    bindEvents();
    syncUrl(false);
    /* skipData：带 ?path= 时数据由下面的 loadPath 统一加载，避免同一端点被请求两次 */
    activateTab(state.activeTab, { skipUrl: true, skipData: true });
    renderSpectrumLegend();
    renderAll();
    loadHealth();
    loadCases();
    if (state.path) {
      state.autoPicked = true;
      loadPath(state.path, { push: false });
    }
    /* 幂等：带 ?tab=fec 但没有 path 时不会有 loadPath，这里补一次（各页都有 pending 守卫） */
    ensureTabData(state.activeTab);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}());
