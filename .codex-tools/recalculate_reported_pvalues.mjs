const tables = {
  table5: {
    labels: [
      'R2 Mean-Q3','R2 Mean-Q3+','R2 Q3-Q3+',
      'RMSE Mean-Q3','RMSE Mean-Q3+','RMSE Q3-Q3+',
      'MAPE Mean-Q3','MAPE Mean-Q3+','MAPE Q3-Q3+',
      'RMSE>Q3 Mean-Q3','RMSE>Q3 Mean-Q3+','RMSE>Q3 Q3-Q3+',
      'RMSE>Q3+ Mean-Q3','RMSE>Q3+ Mean-Q3+','RMSE>Q3+ Q3-Q3+'
    ],
    p: [1.19e-6,5.38e-3,1.46e-1,1.19e-6,7.11e-3,1.40e-1,1.86e-9,3.15e-7,2.05e-1,3.79e-6,1.60e-5,6.26e-1,1.19e-6,3.45e-5,7.77e-1]
  },
  table6: {
    labels: [
      'R2 C-B','R2 C-A','R2 B-A',
      'RMSE C-B','RMSE C-A','RMSE B-A',
      'MAPE C-B','MAPE C-A','MAPE B-A',
      'RMSE>Q3 C-B','RMSE>Q3 C-A','RMSE>Q3 B-A',
      'RMSE>Q3+ C-B','RMSE>Q3+ C-A','RMSE>Q3+ B-A'
    ],
    p: [1.58e-1,2.99e-3,1.23e-3,1.46e-1,1.13e-2,2.196e-3,2.55e-7,7.98e-4,6.19e-3,2.93e-2,1.52e-1,4.27e-2,3.22e-3,1.14e-1,2.34e-2]
  }
};

function holm(ps) {
  const order = ps.map((p,i)=>({p,i})).sort((a,b)=>a.p-b.p);
  const adj = Array(ps.length);
  let running = 0;
  order.forEach((o,k)=>{
    running = Math.max(running, (ps.length-k)*o.p);
    adj[o.i] = Math.min(1,running);
  });
  return adj;
}
for (const [name,t] of Object.entries(tables)) {
  const adjusted = holm(t.p);
  console.log(name);
  t.labels.forEach((label,i)=>console.log(`${label}\traw=${t.p[i]}\tholm15=${adjusted[i]}\tsig=${adjusted[i] < 0.05}`));
}
console.log(`FWER nominal for 15 independent tests: ${1-Math.pow(0.95,15)}`);
console.log(`FWER nominal for 30 independent tests: ${1-Math.pow(0.95,30)}`);
