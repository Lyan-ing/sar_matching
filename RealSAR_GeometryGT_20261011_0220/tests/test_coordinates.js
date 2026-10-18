// Exact same browser coordinate helpers; simulator only, not GT.
const assert=require('assert'),C=require('../annotation_app/coordinates.js');
let cases=0;
for(const width of [128,256,512,1024])for(const scale of [.25,1,2,8,16])for(const offset of [-103.5,0,29]){
 const camera={scale,tx:offset,ty:offset/2};
 for(const p of [[0,0],[width-1,width-1],[30.125,40.75]]){
  const s=C.pixelToScreen(...p,camera),q=C.screenToPixel(...s,camera);assert(Math.max(...q.map((v,i)=>Math.abs(v-p[i])))<1e-9);cases++;
 }
 const anchor=[170.5,310.25],zoomed=C.zoomAt(camera,...anchor,1.3);assert(Math.max(...C.screenToPixel(...anchor,camera).map((v,i)=>Math.abs(v-C.screenToPixel(...anchor,zoomed)[i])))<1e-9);
}
assert.deepStrictEqual(C.clientToCanvas(150,250,{left:50,top:50,width:256,height:512},512,1024),[200,400]);
console.log(JSON.stringify({coordinate_roundtrips:cases,css_scale_zoom_pan_and_pixel_centers:'passed',simulated_only:true}));
