// 자바스크립트 CNN 계산이 파이토치 결과와 같은지 확인: node tools/verify.js
const fs=require('fs'); process.chdir(__dirname + '/..'); global.window={}; global.atob=(s)=>Buffer.from(s,'base64').toString('binary');
eval(fs.readFileSync('src/js/model-weights.js','utf8')); eval(fs.readFileSync('src/js/model.js','utf8'));
eval(fs.readFileSync('src/js/knn-samples.js','utf8')); eval(fs.readFileSync('src/js/knn.js','utf8'));
const ref=JSON.parse(fs.readFileSync('tools/reference.json'));
let maxd=0;
ref.images.forEach((b,i)=>{const raw=Buffer.from(b,'base64');const x=new Float32Array(784);for(let j=0;j<784;j++)x[j]=raw[j]/255;
 const t=Date.now(); const r=window.CNN.predict(x); const ms=Date.now()-t;
 r.probs.forEach((p,d)=>maxd=Math.max(maxd,Math.abs(p-ref.probs[i][d])));
 console.log('img',i,'pred',r.probs.indexOf(Math.max(...r.probs)),'knn',window.KNN.classify(x).label,ms+'ms');});
console.log('max prob diff',maxd);
