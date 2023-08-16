import drawsvg as draw

class ImageProcessor:
    def __init__(self, image, svggroup=None):
        self.image = image
        self.group = svggroup

    def removeRootCluster(self):
        for el in self.group.allChildren():
            if isinstance(el, draw.elements.Path) == True:
                if str(el.id).startswith('rgi'):
                    print(el)

    def moveSampleBox(self, moveX, moveY):
        self.group.args['transform'] = 'translate(' + str(moveX) + ',' + str(moveY) + ')'
        for el in self.group.all_children():
            if isinstance(el, draw.elements.Path) == True:
                if str(el.id).startswith('tnt'):
                    args = el.args['d'].split(' ')
                    M = args[0].split(',')
                    C = args[1].split(',')
                    print(el.args['d'])
                    Mx = float(M[0][1:]) - moveX
                    My = float(M[1]) - moveY

                    # TODO: calculate all new bezier points from new x and y OR maybe better use this function to create boxes and tentacles at first place
                    Cex = float(C[4])
                    Cey = float(C[5])

                    if My > 0:
                        bz1sty = My + 50
                    else:
                        bz1sty = My - 50

                    if My > 0:
                        bz2ndy = Cey + 50
                    else:
                        bz2ndy = Cey - 50

                    bz1stx = (Mx + moveX)
                    bz2ndx = (Cex + moveX)

                    newArgs = "M" + str(Mx) + "," + str(My) + " " + str(bz1stx) + "," + str(bz1sty) + "," + str(
                        bz2ndx) + "," + str(bz2ndy) + "," + str(Cex) + "," + str(Cey)
                    el.args['d'] = newArgs
                    print(el.args['d'])

    def extract_point_by_cluster_color(self, sx, ex, sy, ey, color, preserved_range=[range(-1, -1)]):
        pixels = self.image.load()

        for x in range(sx, ex):
            found = False
            firsty = 0
            lasty = 0
            for y in range(sy, ey):  # this row

                # and this row was exchanged
                # r, g, b = pixels[x, y]

                # in case your image has an alpha channel
                r, g, b, a = pixels[x, y]
                cc = f"#{r:02x}{g:02x}{b:02x}"
                for r in preserved_range:
                    if y not in r:
                        if color == cc and found == False:
                            found = True
                            firsty = y
                        else:
                            if color != cc and found == True:
                                lasty = y
                                if lasty - firsty > 3:
                                    found = False
                                    break
                                else:
                                    found == False

        return firsty, lasty


def add_axes(el):
    for t in range(1, 10):
        l = draw.Line(0, t * 50, 5, t * 50, stroke="black", stroke_width='2')
        el.append(l)
        te = {
            'text': str(t * 50),
            'fontSize': '10',
            'fill': 'black',
            'x': 10,
            'y': t * 50
        }
        # rg.append(draw.Use('rc', 100,100))
        el.append(draw.Text(**te, font_size=12))

    for f in range(1, 20):
        el.append(draw.Line(f * 50, 0, f * 50, 5, stroke="black", stroke_width='2'))
        ty = {
            'text': str(f * 50),
            'fontSize': '10',
            'fill': 'black',
            'x': f * 50,
            'y': 10
        }
        # rg.append(draw.Use('rc', 100,100))
        el.append(draw.Text(**ty, font_size=12))