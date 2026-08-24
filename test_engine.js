const fs = require('fs');
global.window = global;
global.fetch = async (path) => {
  const text = fs.readFileSync('frontend/german_credit.csv', 'utf8');
  return { text: async () => text };
};
require('./frontend/engine.js');

(async () => {
  const api = global.RiskEngineLocal;
  const j = async (r) => await r.json();
  let r;
  r = await j(await api.handle('GET', '/api/v1/data/load-sample'));
  console.log('load-sample:', r.status, 'rows=', r.summary.total_samples, 'features=', r.summary.num_features);

  r = await j(await api.handle('GET', '/api/v1/binning/run?max_bins=5'));
  console.log('binning selected:', r.selected_features.length, 'features');

  r = await j(await api.handle('GET', '/api/v1/training/run'));
  const tr = r.result;
  console.log('train KS=', tr.train_metrics.ks, 'AUC=', tr.train_metrics.auc);
  console.log('test  KS=', tr.test_metrics.ks, 'AUC=', tr.test_metrics.auc);
  console.log('scorecard bins:', tr._scorecard.length);
  console.log('intercept=', tr.intercept.toFixed(4), 'factor=', tr.factor.toFixed(4));

  r = await j(await api.handle('GET', '/api/v1/model/info'));
  console.log('model_info:', JSON.stringify(r));

  r = await j(await api.handle('GET', '/api/v1/explain/sample/0'));
  console.log('sample0 score=', r.score, 'decision=', r.decision, 'factors=', r.breakdown.length);

  r = await j(await api.handle('GET', '/api/v1/explain/importance'));
  console.log('top importance:', r.importance.slice(0,3).map(x=>x.feature+':'+x.importance_pct+'%').join(', '));

  r = await j(await api.handle('GET', '/api/v1/explain/algorithm'));
  console.log('algorithm selected:', r.selected_algorithm, 'comparisons=', r.comparison.length);

  r = await j(await api.handle('GET', '/api/v1/explain/audit'));
  console.log('audit overall:', r.overall_status, 'passed=', r.passed, '/', r.total_checks);

  r = await (await api.handle('GET', '/api/v1/export/python')).text();
  console.log('export python length:', r.length);

  r = await (await api.handle('GET', '/api/v1/export/html')).text();
  console.log('export html length:', r.length);

  console.log('ALL_OK');
})().catch(e => { console.error('ERROR', e); process.exit(1); });
