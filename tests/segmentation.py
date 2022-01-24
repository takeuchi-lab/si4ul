from si4ul import segmentation_si

lw = segmentation_si.th('./tests/image/liver1cut.jpg', './tests/image/result')
segmentation_si.psegi_th(lw)