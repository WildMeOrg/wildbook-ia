from os.path import join, dirname, realpath, expanduser, exists
from ibeis.control import IBEISControl
from ibeis.dev import ibsfuncs
from detecttools.directory import Directory

# prefix = dirname(realpath(__file__))
prefix = expanduser(join("~", "Desktop"))

print "PROCESSING ACTIVITIES"
activities = {}
columns = [3, 9, 10, 11, 12, 13, 14, 15]
for line in open(join(prefix, 'Group-Habitat-Activity table.csv'), 'r').readlines()[1:]:
    line = [ item.strip() for item in line.strip().split(',')]
    _id = line[2]
    if _id not in activities:
        activities[_id] = [line[col] for col in columns]

# FIX FILES DIRECTORY
originals = join(prefix, 'originals')
images = Directory(originals)
exts = []
for image in images.files():
    exts.append(image.split(".")[-1])
exts = list(set(exts))
print "EXTENSIONS:", exts

print "PROCESSING ENCOUNTERS"
used = []
encounters = open(join(prefix, 'encounters.csv'),'w')
animals = open(join(prefix, 'animals.csv'),'w')
linenum = 0
processed = []
for line in open(join(prefix, 'Individual sightings.csv'), 'r').readline().split('\r')[1:]:
    linenum += 1
    line = [ item.strip() for item in line.strip().split(',')]
    if len(line) == 1:
        print "INVALID DATA ON LINE", linenum, "[FIX TO CONTINUE]"
        break 
    filename = line[2]
    sighting = line[1]
    files = [ join(originals, filename + "." + ext) in images.files() for ext in exts]

    if sighting in activities and True in files:
        for i in range(len(files)):
            if files[i]:
                filename += "." + exts[i]
                break

        line = [join(originals, filename)] + activities[sighting]
        if filename not in used:
            processed.append(line)
            animals.write(",".join(line) + "\n")
            used.append(filename)
        encounters.write(",".join(line) + "\n")

print "USED:", float(len(used)) / len(images.files())

dbdir = join(prefix, "converted")
ibsfuncs.delete_ibeis_database(dbdir)
ibs = IBEISControl.IBEISController(dbdir=dbdir)
image_gpath_list = [item[0] for item in processed]
assert all(map(exists, image_gpath_list)), 'some images dont exist'

gid_list = ibs.add_images(image_gpath_list)
bbox_list = [ (0, 0, w, h) for (w, h) in ibs.get_image_sizes(gid_list) ]
ibs.add_rois(gid_list, bbox_list)
ibs.db.commit()
ibs.db.dump()