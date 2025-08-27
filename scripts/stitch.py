import argparse, glob, os, cv2, numpy as np

def stitch_images(img_paths):
    imgs = [cv2.imread(p) for p in img_paths]
    mid = len(imgs)//2
    base = imgs[mid]
    orb = cv2.ORB_create(4000)
    def kps_desc(img): return orb.detectAndCompute(img, None)
    def warp_into(base, img):
        kpsA, desA = kps_desc(base); kpsB, desB = kps_desc(img)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desB, desA, k=2)
        good = [m for m,n in matches if m.distance < 0.75*n.distance]
        if len(good) < 8: return None
        ptsA = np.float32([kpsA[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        ptsB = np.float32([kpsB[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        H, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
        hA,wA = base.shape[:2]; hB,wB = img.shape[:2]
        corners = np.float32([[0,0],[wB,0],[wB,hB],[0,hB]]).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(corners, H)
        [xmin, ymin] = np.int32(warped.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(warped.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]; Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
        result = cv2.warpPerspective(img, Ht@H, (xmax-xmin, ymax-ymin))
        result[t[1]:hA+t[1], t[0]:wA+t[0]] = base
        return result
    canvas = base
    for i in range(mid-1, -1, -1):
        out = warp_into(canvas, imgs[i])
        if out is not None: canvas = out
    for i in range(mid+1, len(imgs)):
        out = warp_into(canvas, imgs[i])
        if out is not None: canvas = out
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--save", default="outputs/panorama.jpg")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.folder, args.glob)))
    assert len(paths) >= 2, "Need at least two images"
    pano = stitch_images(paths)
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(args.save, pano)
    print(f"Saved: {args.save}")

if __name__ == "__main__":
    main()
