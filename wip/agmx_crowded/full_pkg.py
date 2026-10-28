"""label image of the full new package (v3 background + crossing split + serial streaks + linking) -> out/IMG_EXT_pkg.npz"""
import sys, numpy as np
sys.path.insert(0, "."); import harness as HX, bgv3, xsplit, streaks, muons2, muons3
if __name__ == "__main__":
    img, ext = int(sys.argv[1]), int(sys.argv[2])
    Q, valid = HX.load(img, ext)
    bg, rep = bgv3.regional_v3(Q, valid, HX.AM.background_map(Q, valid))
    var = sys.argv[3] if len(sys.argv) > 3 else "pkg"
    xsplit.install(HX.GX); streaks.install(HX.AD)
    if var == "pkgmu2": muons2.install(HX.AD)
    if var == "pkgmu3": muons3.install(HX.AD)
    cl = [c for c in HX.AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True) if "_pix" in c and np.isfinite(c.get("E", np.nan))]
    cl, nl = HX.link_tracks(cl)
    lab = np.zeros(Q.shape, np.int32); isstreak = []
    for k, c in enumerate(cl, 1):
        ys, xs, q = c["_pix"]; lab[ys, xs] = k; isstreak.append(c.get("regime") == "serial")
    tab = {k: np.array([c.get(k, np.nan) for c in cl], float) for k in ("E", "x", "y", "sxy", "blend", "ped")}
    np.savez_compressed(f"out/{img}_{ext}_{var}.npz", lab=lab, cls=np.array([c.get("cls", "") for c in cl]), **{k: v for k, v in tab.items() if k != "E"}, npix=np.array([c["_pix"][0].size for c in cl]), streak=np.array(isstreak), E=np.array([c["E"] for c in cl]))
    print(img, ext, len(cl), "clusters,", int(np.sum(isstreak)), "streaks,", nl, "linked")
