// Zero-based pixel centers. CSS rect scale and zoom are independent.
(function(root){
  const api={
    screenToPixel:(x,y,c)=>[(x-c.tx)/c.scale-.5,(y-c.ty)/c.scale-.5],
    pixelToScreen:(x,y,c)=>[(x+.5)*c.scale+c.tx,(y+.5)*c.scale+c.ty],
    clientToCanvas:(x,y,r,w,h)=>[(x-r.left)*w/r.width,(y-r.top)*h/r.height],
    zoomAt:(c,x,y,f)=>{const p=api.screenToPixel(x,y,c),s=Math.max(.25,Math.min(16,c.scale*f));return {scale:s,tx:x-(p[0]+.5)*s,ty:y-(p[1]+.5)*s};}
  };if(typeof module!=='undefined')module.exports=api;else root.Coordinates=api;
})(typeof window!=='undefined'?window:globalThis);
