"""Reference-only search regions, never synthetic human correspondences."""
from experiment import *
from PIL import ImageDraw
def main():
    out=ROOT/'review_guide';out.mkdir(exist_ok=False);tasks=lines(ROOT/'annotation_adapter/annotation_tasks/tasks.jsonl');regions=[];text=['# 五对独立核验任务','', '默认隐藏自动对应与地图对应。先在原生图独立点击静态且双图可辨认的地物，由另一名人员独立复核并记录不确定度。主动显示辅助层会永久记录暴露；不能将辅助点升级为独立C证据。','', '下列位置仅是参考图的检索区域，由参考图局部方差和空间分散选择，不是对应点，也未确认静态地物或物理散射中心。码头上的舰船、货物和海面亮点不能默认静态。','']
    for task in tasks:
        sid=task['sample_id'];a=np.array(Image.open(task['image0']));b=np.array(Image.open(task['image1']));blur=cv2.GaussianBlur(a.astype(np.float32),(0,0),2)
        candidates=sorted([(float(blur[y:y+96,x:x+96].std()),x,y) for y in [32,160,288,384] for x in [32,160,288,384]],reverse=True);chosen=[]
        for score,x,y in candidates:
            if all(np.hypot(x-p['x'],y-p['y'])>=160 for p in chosen):chosen.append({'x':x,'y':y,'width':96,'height':96,'reference_only_std':score,'static_or_correspondence_verified':False})
            if len(chosen)==3:break
        canvas=Image.new('RGB',(1024,590),'white');canvas.paste(Image.fromarray(a).convert('RGB'),(0,55));canvas.paste(Image.fromarray(b).convert('RGB'),(512,55));draw=ImageDraw.Draw(canvas);draw.text((4,4),sid+' | '+task['look_pair_category'],fill='black');draw.text((4,22),'Reference-only search ROIs; NO verified static points or moving correspondences. Native PNG.',fill='black')
        for i,p in enumerate(chosen):draw.rectangle((p['x'],p['y']+55,p['x']+96,p['y']+151),outline='#E69F00',width=2);draw.text((p['x']+3,p['y']+57),str(i+1),fill='yellow')
        draw.text((3,573),'Umbra / Copyright Capella Space CC BY4; Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery, 2026',fill='black');canvas.save(out/f'{sid}.png');regions.append({'sample_id':sid,'look_pair_category':task['look_pair_category'],'selection_reason':task['selection_reason'],'reference_search_ROIs':chosen,'NO_automatic_human_points':True})
        text.extend([f'## {sid}', '',f"{task['site_name']}；{task['look_pair_category']}。{task['selection_reason']}", '',f'参考图检索ROI：{[(p["x"],p["y"],96,96) for p in chosen]}。先确认固定建筑/岸线设施/道路交叉等，不能仅凭相似亮点。', '',f'预览：[{sid}.png]({sid}.png)', ''])
    save(out/'search_ROIs.json',regions);(out/'README.md').write_text('\n'.join(text));print('five independent review guides; no correspondences generated')
if __name__=='__main__':main()
