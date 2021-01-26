"""
ade20k dataloaders
"""

import os
import glob
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


S2I = {
    "airfield": 0,
    "airplane_cabin": 1,
    "airport_terminal": 2,
    "alcove": 3,
    "alley": 4,
    "amphitheater": 5,
    "amusement_arcade": 6,
    "amusement_park": 7,
    "apartment_building-outdoor": 8,
    "aquarium": 9,
    "aqueduct": 10,
    "arcade": 11,
    "arch": 12,
    "archaelogical_excavation": 13,
    "archive": 14,
    "arena-hockey": 15,
    "arena-performance": 16,
    "arena-rodeo": 17,
    "army_base": 18,
    "art_gallery": 19,
    "art_school": 20,
    "art_studio": 21,
    "artists_loft": 22,
    "assembly_line": 23,
    "athletic_field-outdoor": 24,
    "atrium-public": 25,
    "attic": 26,
    "auditorium": 27,
    "auto_factory": 28,
    "auto_showroom": 29,
    "badlands": 30,
    "bakery-shop": 31,
    "balcony-exterior": 32,
    "balcony-interior": 33,
    "ball_pit": 34,
    "ballroom": 35,
    "bamboo_forest": 36,
    "bank_vault": 37,
    "banquet_hall": 38,
    "bar": 39,
    "barn": 40,
    "barndoor": 41,
    "baseball_field": 42,
    "basement": 43,
    "basketball_court-indoor": 44,
    "bathroom": 45,
    "bazaar-indoor": 46,
    "bazaar-outdoor": 47,
    "beach": 48,
    "beach_house": 49,
    "beauty_salon": 50,
    "bedchamber": 51,
    "bedroom": 52,
    "beer_garden": 53,
    "beer_hall": 54,
    "berth": 55,
    "biology_laboratory": 56,
    "boardwalk": 57,
    "boat_deck": 58,
    "boathouse": 59,
    "bookstore": 60,
    "booth-indoor": 61,
    "botanical_garden": 62,
    "bow_window-indoor": 63,
    "bowling_alley": 64,
    "boxing_ring": 65,
    "bridge": 66,
    "building_facade": 67,
    "bullring": 68,
    "burial_chamber": 69,
    "bus_interior": 70,
    "bus_station-indoor": 71,
    "butchers_shop": 72,
    "butte": 73,
    "cabin-outdoor": 74,
    "cafeteria": 75,
    "campsite": 76,
    "campus": 77,
    "canal-natural": 78,
    "canal-urban": 79,
    "candy_store": 80,
    "canyon": 81,
    "car_interior": 82,
    "carrousel": 83,
    "castle": 84,
    "catacomb": 85,
    "cemetery": 86,
    "chalet": 87,
    "chemistry_lab": 88,
    "childs_room": 89,
    "church-indoor": 90,
    "church-outdoor": 91,
    "classroom": 92,
    "clean_room": 93,
    "cliff": 94,
    "closet": 95,
    "clothing_store": 96,
    "coast": 97,
    "cockpit": 98,
    "coffee_shop": 99,
    "computer_room": 100,
    "conference_center": 101,
    "conference_room": 102,
    "construction_site": 103,
    "corn_field": 104,
    "corral": 105,
    "corridor": 106,
    "cottage": 107,
    "courthouse": 108,
    "courtyard": 109,
    "creek": 110,
    "crevasse": 111,
    "crosswalk": 112,
    "dam": 113,
    "delicatessen": 114,
    "department_store": 115,
    "desert-sand": 116,
    "desert-vegetation": 117,
    "desert_road": 118,
    "diner-outdoor": 119,
    "dining_hall": 120,
    "dining_room": 121,
    "discotheque": 122,
    "doorway-outdoor": 123,
    "dorm_room": 124,
    "downtown": 125,
    "dressing_room": 126,
    "driveway": 127,
    "drugstore": 128,
    "elevator-door": 129,
    "elevator_lobby": 130,
    "elevator_shaft": 131,
    "embassy": 132,
    "engine_room": 133,
    "entrance_hall": 134,
    "escalator-indoor": 135,
    "excavation": 136,
    "fabric_store": 137,
    "farm": 138,
    "fastfood_restaurant": 139,
    "field-cultivated": 140,
    "field-wild": 141,
    "field_road": 142,
    "fire_escape": 143,
    "fire_station": 144,
    "fishpond": 145,
    "flea_market-indoor": 146,
    "florist_shop-indoor": 147,
    "food_court": 148,
    "football_field": 149,
    "forest-broadleaf": 150,
    "forest_path": 151,
    "forest_road": 152,
    "formal_garden": 153,
    "fountain": 154,
    "galley": 155,
    "garage-indoor": 156,
    "garage-outdoor": 157,
    "gas_station": 158,
    "gazebo-exterior": 159,
    "general_store-indoor": 160,
    "general_store-outdoor": 161,
    "gift_shop": 162,
    "glacier": 163,
    "golf_course": 164,
    "greenhouse-indoor": 165,
    "greenhouse-outdoor": 166,
    "grotto": 167,
    "gymnasium-indoor": 168,
    "hangar-indoor": 169,
    "hangar-outdoor": 170,
    "harbor": 171,
    "hardware_store": 172,
    "hayfield": 173,
    "heliport": 174,
    "highway": 175,
    "home_office": 176,
    "home_theater": 177,
    "hospital": 178,
    "hospital_room": 179,
    "hot_spring": 180,
    "hotel-outdoor": 181,
    "hotel_room": 182,
    "house": 183,
    "hunting_lodge-outdoor": 184,
    "ice_cream_parlor": 185,
    "ice_floe": 186,
    "ice_shelf": 187,
    "ice_skating_rink-indoor": 188,
    "ice_skating_rink-outdoor": 189,
    "iceberg": 190,
    "igloo": 191,
    "industrial_area": 192,
    "inn-outdoor": 193,
    "islet": 194,
    "jacuzzi-indoor": 195,
    "jail_cell": 196,
    "japanese_garden": 197,
    "jewelry_shop": 198,
    "junkyard": 199,
    "kasbah": 200,
    "kennel-outdoor": 201,
    "kindergarden_classroom": 202,
    "kitchen": 203,
    "lagoon": 204,
    "lake-natural": 205,
    "landfill": 206,
    "landing_deck": 207,
    "laundromat": 208,
    "lawn": 209,
    "lecture_room": 210,
    "legislative_chamber": 211,
    "library-indoor": 212,
    "library-outdoor": 213,
    "lighthouse": 214,
    "living_room": 215,
    "loading_dock": 216,
    "lobby": 217,
    "lock_chamber": 218,
    "locker_room": 219,
    "mansion": 220,
    "manufactured_home": 221,
    "market-indoor": 222,
    "market-outdoor": 223,
    "marsh": 224,
    "martial_arts_gym": 225,
    "mausoleum": 226,
    "medina": 227,
    "mezzanine": 228,
    "moat-water": 229,
    "mosque-outdoor": 230,
    "motel": 231,
    "mountain": 232,
    "mountain_path": 233,
    "mountain_snowy": 234,
    "movie_theater-indoor": 235,
    "museum-indoor": 236,
    "museum-outdoor": 237,
    "music_studio": 238,
    "natural_history_museum": 239,
    "nursery": 240,
    "nursing_home": 241,
    "oast_house": 242,
    "ocean": 243,
    "office": 244,
    "office_building": 245,
    "office_cubicles": 246,
    "oilrig": 247,
    "operating_room": 248,
    "orchard": 249,
    "orchestra_pit": 250,
    "pagoda": 251,
    "palace": 252,
    "pantry": 253,
    "park": 254,
    "parking_garage-indoor": 255,
    "parking_garage-outdoor": 256,
    "parking_lot": 257,
    "pasture": 258,
    "patio": 259,
    "pavilion": 260,
    "pet_shop": 261,
    "pharmacy": 262,
    "phone_booth": 263,
    "physics_laboratory": 264,
    "picnic_area": 265,
    "pier": 266,
    "pizzeria": 267,
    "playground": 268,
    "playroom": 269,
    "plaza": 270,
    "pond": 271,
    "porch": 272,
    "promenade": 273,
    "pub-indoor": 274,
    "racecourse": 275,
    "raceway": 276,
    "raft": 277,
    "railroad_track": 278,
    "rainforest": 279,
    "reception": 280,
    "recreation_room": 281,
    "repair_shop": 282,
    "residential_neighborhood": 283,
    "restaurant": 284,
    "restaurant_kitchen": 285,
    "restaurant_patio": 286,
    "rice_paddy": 287,
    "river": 288,
    "rock_arch": 289,
    "roof_garden": 290,
    "rope_bridge": 291,
    "ruin": 292,
    "runway": 293,
    "sandbox": 294,
    "sauna": 295,
    "schoolhouse": 296,
    "science_museum": 297,
    "server_room": 298,
    "shed": 299,
    "shoe_shop": 300,
    "shopfront": 301,
    "shopping_mall-indoor": 302,
    "shower": 303,
    "ski_resort": 304,
    "ski_slope": 305,
    "sky": 306,
    "skyscraper": 307,
    "slum": 308,
    "snowfield": 309,
    "soccer_field": 310,
    "stable": 311,
    "stadium-baseball": 312,
    "stadium-football": 313,
    "stadium-soccer": 314,
    "stage-indoor": 315,
    "stage-outdoor": 316,
    "staircase": 317,
    "storage_room": 318,
    "street": 319,
    "subway_station-platform": 320,
    "supermarket": 321,
    "sushi_bar": 322,
    "swamp": 323,
    "swimming_hole": 324,
    "swimming_pool-indoor": 325,
    "swimming_pool-outdoor": 326,
    "synagogue-outdoor": 327,
    "television_room": 328,
    "television_studio": 329,
    "temple-asia": 330,  # FIXME Shouldn't this be temple-east-asia?
    "throne_room": 331,
    "ticket_booth": 332,
    "topiary_garden": 333,
    "tower": 334,
    "toyshop": 335,
    "train_interior": 336,
    "train_station-platform": 337,
    "tree_farm": 338,
    "tree_house": 339,
    "trench": 340,
    "tundra": 341,
    "underwater-ocean_deep": 342,
    "utility_room": 343,
    "valley": 344,
    "vegetable_garden": 345,
    "veterinarians_office": 346,
    "viaduct": 347,
    "village": 348,
    "vineyard": 349,
    "volcano": 350,
    "volleyball_court-outdoor": 351,
    "waiting_room": 352,
    "water_park": 353,
    "water_tower": 354,
    "waterfall": 355,
    "watering_hole": 356,
    "wave": 357,
    "wet_bar": 358,
    "wheat_field": 359,
    "wind_farm": 360,
    "windmill": 361,
    "yard": 362,
    "youth_hostel": 363,
    "zen_garden": 364,
}
S2I = {f"{k}-s": v for k, v in S2I.items()}

I2S = {v: k for k, v in S2I.items()}


def to_dataloader(dataset, batch_size=32, shuffle=True, **kwargs):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, **kwargs)


transformtypedict = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color,
)


class ImageJitter:
    def __init__(self, transformdict):
        self.transforms = [
            (transformtypedict[k], transformdict[k]) for k in transformdict
        ]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")

        return out


class TransformLoader:
    def __init__(
        self,
        image_size,
        normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4),
    ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == "ImageJitter":
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == "RandomResizedCrop":
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            return method(self.image_size)
        elif transform_type == "Resize":
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == "Normalize":
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False, normalize=True, to_pil=True):
        if aug:
            transform_list = [
                "RandomResizedCrop",
                "ImageJitter",
                "RandomHorizontalFlip",
                "ToTensor",
            ]
        else:
            transform_list = ["Resize", "CenterCrop", "ToTensor"]

        if normalize:
            transform_list.append("Normalize")

        if to_pil:
            transform_list = ["ToPILImage"] + transform_list

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    def get_normalize(self):
        return self.parse_transform("Normalize")


class ADE20K:
    def __init__(self, path, split, max_classes=None):
        assert split in {'training', 'validation'}
        self.split = split
        self.path = path
        self.img_dir = os.path.join(self.path, 'images', self.split)
        self.max_classes = max_classes

        # Transform
        self.transform_loader = TransformLoader(224)
        augment = self.split == 'training'
        self.transform = self.transform_loader.get_composed_transform(augment, to_pil=False)

        # Load images
        self.images = []
        self.classes = []

        self.load_images()
        self.unique_classes = np.unique(self.classes)
        self.n_classes = 365

    def load_images(self):
        n_classes = 0
        for letterdir in os.listdir(self.img_dir):
            if len(letterdir) != 1:
                continue
            letterdir = os.path.join(self.img_dir, letterdir)
            for cname in os.listdir(letterdir):
                cname_dir = os.path.join(letterdir, cname)
                has_nested_dirs = any(os.path.isdir(os.path.join(cname_dir, i)) for i in os.listdir(cname_dir))
                subcnames = []
                if has_nested_dirs:
                    # Go one deeper
                    for subcname in os.listdir(cname_dir):
                        assert os.path.isdir(os.path.join(cname_dir, subcname))
                        subcnames.append(subcname)
                else:
                    subcnames.append('')

                for subcname in subcnames:
                    if subcname != '':
                        # Some exceptions (classes that are ignored in places365)
                        # car_interior-s, orchestra_pit-s, soccer_field-s, temple-asia-s, waterfall-s
                        # orchestra pit and soccer field I couldn't find
                        if cname == 'waterfall':
                            combname = 'waterfall-s'
                        elif cname == 'car_interior':
                            combname = 'car_interior-s'
                        elif cname == 'temple' and subcname == 'east_asia':
                            combname = 'temple-asia-s'
                        else:
                            combname = f"{cname}-{subcname}-s"
                        subcnamedir = os.path.join(cname_dir, subcname)
                    else:
                        combname = f"{cname}-s"
                        subcnamedir = cname_dir
                    if combname not in S2I:
                        continue
                    for imgf in glob.glob(os.path.join(subcnamedir, '*.jpg')):
                        self.images.append(imgf)
                        self.classes.append(S2I[combname])
                    n_classes += 1
                    if self.max_classes is not None and n_classes >= self.max_classes:
                        return

    def __getitem__(self, i):
        img_path = self.images[i]
        cl = self.classes[i]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, cl

    def __len__(self):
        return len(self.images)


def load_ade20k(path, max_classes=None, random_state=None):
    return {
        'train': ADE20K(path, 'training', max_classes=max_classes),
        'val': ADE20K(path, 'validation', max_classes=max_classes),
    }
