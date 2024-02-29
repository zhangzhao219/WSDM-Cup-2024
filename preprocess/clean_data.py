import re
import emoji
import json
from tqdm import tqdm

emoji_list_drop = []
emoji_list_reserved = []

# 需要完全保留
emoji_reserved_list = [
    "[CTRL+ENTER]",
    "[China Mobile High Frequency Harassment Phone H Protection]",
    "[HDR mode]",
    "[Scan the QR code]",
]

# 需要指定替换的方式
emoji_reserved_replace_dict = {
    "[How to preliminarily judge whether there has been metastasis through the examination report? See figure [ThreeR]": "[How to preliminarily judge whether there has been metastasis through the examination report? See figure Three",
    "[I also found it for a while[covering face R]": "[I also found it for a while",
    "[Just kidding[SmileR]": "[Just kidding",
    "[No, anyway, I didn't see it [sneaky R]": "[No, anyway, I didn't see it. ",
    "[The old device QR code won't be pushed off, feel at ease]": "[The old device QR code won't be pushed off, feel at ease.",
    "[ps: The cat head indicates that the packaging of the trash can is also very nice~ Just for this, it's worth buying more! [Big Laugh R]": "[ps: The cat head indicates that the packaging of the trash can is also very nice~ Just for this, it's worth buying more!",
    "[】 Decrease font size [Right R]": "[】 Decrease font size ",
}

# 需要加上反括号后替换
emoji_reserved_replace_plus_dict = {
    "[[Black potato question mark R]": " ",
    "[[Red heart shaped R]": " ",
    "[[Rose R]": " ",
}

# 需要丢弃的情感符号
emoji_drop_list = [
    "[破涕为笑]",
    "[话题]",
    "[wink]",
    "[yeah]",
    "[yea]",
    "[ye]",
    "[wow]",
    "[weeping]",
    "[wa]",
    "[victory]",
    "[thumb up]",
    "[thumbs up]",
    "[time]",
    "[teeth]",
    "[tears]",
    "[tearing into a smile]",
    "[t]",
    "[sweat]",
    "[sweating]",
    "[support wall]",
    "[super cheap]",
    "[sun]",
    "[strong]",
    "[stone]",
    "[star]",
    "[stealth observation]",
    "[stealing a laugh]",
    "[sneer]",
    "[sneaky]",
    "[sneaky smile]",
    "[sneaky laughter]",
    "[sneaky laugh]",
    "[smug]",
    "[smiling]",
    "[smiling face]",
    "[smile]",
    "[smart]",
    "[smack]",
    "[sly]",
    "[sighing]",
    "[sigh]",
    "[shy]",
    "[shy gesture]",
    "[shrug]",
    "[shocked]",
    "[selfie]",
    "[seduce]",
    "[rose]",
    "[scared]",
    "[right]",
    "[right arrow]",
    "[proud]",
    "[praising]",
    "[poor]",
    "[planting grass]",
    "[plant grass]",
    "[outside]",
    "[observing secretly]",
    "[observe]",
    "[observant]",
    "[moon]",
    "[love]",
    "[love heart]",
    "[like]",
    "[let me see]",
    "[laughs]",
    "[laughs and cries]",
    "[laughing]",
    "[laughing to tears]",
    "[laughing crying]",
    "[laughing cry]",
    "[laughing and crying]",
    "[laugh]",
    "[laugh/cry]",
    "[laugh cry]",
    "[laugh and cry]",
    "[kissing]",
    "[kissing gesture]",
    "[kissing face]",
    "[kiss]",
    "[ill]",
    "[hey ha]",
    "[honey]",
    "[horror]",
    "[happy]",
    "[handshake]",
    "[grinning]",
    "[grinning face]",
    "[grin]",
    "[grimace]",
    "[good]",
    "[frowning]",
    "[frown]",
    "[giggling]",
    "[giggles]",
    "[gift]",
    "[gift   ]",
    "[fist]",
    "[fireworks]",
    "[explosion]",
    "[exhale]",
    "[evil grin]",
    "[embarrassment]",
    "[embarrassed]",
    "[embarrassed face]",
    "[dres]",
    "[doge]",
    "[disappointed]",
    "[disappointed face]",
    "[despise]",
    "[cuteness]",
    "[cute]",
    "[crying]",
    "[crying with laughter]",
    "[crying laughing]",
    "[crying face]",
    "[cry]",
    "[cried]",
    "[crazy]",
    "[crack]",
    "[covers face]",
    "[covering face]",
    "[cover face]",
    "[cough]",
    "[color]",
    "[colorful]",
    "[come on]",
    "[celebration]",
    "[celebrate]",
    "[cheer]",
    "[bite]",
    "[big laugh]",
    "[bad smile]",
    "[ashamed]",
    "[applause]",
    "[angry]",
    "[blowing kiss]",
    "[blowing a kiss]",
    "[  ]",
    "[ ]",
    "[]",
    "[#]",
    "[Angry Emoji]",
    "[Angry]",
    "[ApplauseEmoji]",
    "[Applause]",
    "[Attention!]",
    "[Beauty]",
    "[Big Laugh]",
    "[Big Laughs]",
    "[Broken Heart]",
    "[CRYING]",
    "[CRY]",
    "[Celebrate]",
    "[Celebration]",
    "[Chuckle]",
    "[Chuckles]",
    "[Contempt]",
    "[Cool]",
    "[Crazy]",
    "[Cry]",
    "[Crying Laughing Emoji]",
    "[Crying laughing]",
    "[Crying]",
    "[Curious Emoji]",
    "[Cute Emoji]",
    "[Cute]",
    "[Wow]",
    "[Wronged]",
    "[Wow Emoji]",
    "[Wow face] ",
    "[Wow!]",
    "[WowEmoji]",
    "[Wa]",
    "[WOW]",
    "[When crying]",
    "[Warm tips]",
    "[Warm Tips]",
    "[Victory]",
    "[TwoEmoji]",
    "[Thumb Emoji]",
    "[Thumb Up]",
    "[Thumb up]",
    "[ThumbUpEmoji]",
    "[Thumb]",
    "[Thumbs Up Emoji]",
    "[Thumbs Up] ",
    "[Thumbs up]",
    "[Topic]",
    "[Time]",
    "[Teasing]",
    "[Teasing Emoji]",
    "[Tears]",
    "[Tears into smile]",
    "[Taboo]",
    "[Sweat Face]",
    "[Sweat]",
    "[Sweating]",
    "[Support Wall]",
    "[SunEmoji]",
    "[Sun]",
    "[Strong]",
    "[Star]",
    "[Stealing a laugh]",
    "[Stealing laughter] ",
    "[Stealthy laughter]",
    "[Squinting]",
    "[Squat]",
    "[Squatting]",
    "[Sparkles]",
    "[So sad]",
    "[Sneer]",
    "[Sneaky Emoji]",
    "[Sneaky Laugh Emoji]",
    "[Sneaky Laugh]",
    "[Sneaky Laughter]",
    "[Sneaky laugh]",
    "[Sneaky laughter]",
    "[Sneaky smile]",
    "[Sneaky]",
    "[Smug]",
    "[Smirk]",
    "[Smart]",
    "[SmileEmoji]",
    "[Smile]",
    "[Smiling face with heart-shaped eyes]",
    "[Smack]",
    "[Side Eye Emoji]",
    "[Side eye]",
    "[Side-eye Emoji]",
    "[Side-eye]",
    "[Side-eyed]",
    "[Sigh]",
    "[Shy]",
    "[ShyEmoji]",
    "[Show off]",
    "[Shocked Emoji]",
    "[Shocked]",
    "[See]",
    "[Seduce]",
    "[Scared]",
    "[Sad]",
    "[STONEFACE]",
    "[Rthumb]",
    "[Rose]",
    "[Rkissing]",
    "[RightEmoji]",
    "[Right]",
    "[Rightward Arrow]",
    "[Right Arrow]",
    "[Right arrow]",
    "[RedBookEmoji]",
    "[Red Heart]",
    "[Red heart Emoji]",
    "[Red heart]",
    "[Raising eyebrows]",
    "[Raising Hand]",
    "[Raise]",
    "[Raise hand]",
    "[Raise Hand]",
    "[Raise Hand Emoji]",
    "[Rainbow]",
    "[RUNNING]",
    "[Questioning Emoji]",
    "[Rcrying]",
    "[Question]",
    "[Pulling face]",
    "[Pull Grass]",
    "[Pull grass]",
    "[Pull out the grass]",
    "[Proud]",
    "[Praying Hands]",
    "[Praise]",
    "[Plane]",
    "[Planting Grass]",
    "[Planting grass]",
    "[Poor]",
    "[Pitiful]",
    "[Party Emoji]",
    "[Party]",
    "[OneEmoji]",
    "[Observing in secret]",
    "[Observing in the Dark]",
    "[Observing in the dark]",
    "[Observing secretly]",
    "[ObservingEmoji]",
    "[Observing]",
    "[Naive]",
    "[Miscellaneous]",
    "[Mischievous smile]",
    "[Meng Meng Da]",
    "[Melon]",
    "[Melon eating]",
    "[Love]",
    "[MENG]",
    "[Let me see]",
    "[Laugh Cry]",
    "[Laugh Crying Emoji]",
    "[Laugh Emoji]",
    "[Laugh and cry]",
    "[Laugh]",
    "[Laughing Emoji]",
    "[Laughing Face]",
    "[Laughing and crying]",
    "[Laughing softly]",
    "[Laughing till Tears]",
    "[Laughing to tears]",
    "[Laughing]",
    "[Laughs]",
    "[LOVELY]",
    "[LOL]",
    "[Knife]",
    "[Kiss]",
    "[Kisses]",
    "[Kissing Emoji]",
    "[Joyful]",
    "[Joy]",
    "[Jiong]",
    "[Heart]",
    "[HeartEmoji]",
    "[Happy]",
    "[Handshake]",
    "[Head]",
    "[Hand Raising]",
    "[Hand up]",
    "[Grinning]",
    "[Grin]",
    "[Grinning face]",
    "[Green Heart]",
    "[Great]",
    "[Grass]",
    "[Grass Planting]",
    "[Grass planting]",
    "[Grass-pulling]",
    "[Good]",
    "[Good words]",
    "[Gold]",
    "[Give a hand]",
    "[Girl Heart]",
    "[Giggling]",
    "[Gift]",
    "[FullMoonEmoji]",
    "[Full Moon]",
    "[Frown]",
    "[FourEmoji]",
    "[Fortune]",
    "[FiveEmoji]",
    "[Fireworks]",
    "[Fire]",
    "[FireEmoji]",
    "[Fairy]",
    "[Eye-roll]",
    "[Eye Roll]",
    "[ExplodingEmoji]",
    "[Evil Eyes]",
    "[Evil laughter]",
    "[Emm]",
    "[Emoji of big laugh]",
    "[Emoji of flying kiss]",
    "[Emoji of sneaky laugh]",
    "[Emoji of tongue out]",
    "[Emoji]",
    "[Embarrassed Emoji]",
    "[Embarrassed]",
    "[EMBARRASSED]",
    "[Doubt]",
    "[Doge]",
    "[Dog head]",
    "[Dolphin]",
    "[Disappointed Emoji]",
    "[Disappointed]",
    "[Diamond]",
]


def deal_text(text):
    # 表情符号处理
    no_emoji = emoji.replace_emoji(text, replace=" ")
    # 文字隐含的表情符号
    emoji_list_find = re.findall("\[.*?]", no_emoji)
    # print(emoji_list_find)
    for emoji_str in emoji_list_find:
        is_emoji_sign = False
        if "emoji" in emoji_str:
            is_emoji_sign = True
        elif emoji_str in emoji_reserved_list:
            is_emoji_sign = False
        elif emoji_str in emoji_reserved_replace_dict:
            no_emoji = no_emoji.replace(
                emoji_str, emoji_reserved_replace_dict[emoji_str]
            )
            is_emoji_sign = False
        elif emoji_str in emoji_reserved_replace_plus_dict:
            no_emoji = no_emoji.replace(
                emoji_str + "]", emoji_reserved_replace_plus_dict[emoji_str]
            )
            is_emoji_sign = False
        elif emoji_str in emoji_drop_list:
            is_emoji_sign = True
        elif "R" in emoji_str or "H" in emoji_str:
            for emoji_words in emoji_str[1:-1].split(" "):
                if (
                    emoji_words == "R"
                    or emoji_words == "H"
                    or emoji_words.endswith("R")
                    or emoji_words.endswith("H")
                ):
                    is_emoji_sign = True
                    break
        if not is_emoji_sign:
            emoji_list_reserved.append(emoji_str)
        else:
            emoji_list_drop.append(emoji_str)
            no_emoji = no_emoji.replace(emoji_str, " ")
    # 空格处理
    no_space = no_emoji.replace(" \n", "\n")
    no_space = no_space.replace("\n ", "\n")
    no_space = re.sub("\s+", " ", no_space)
    return no_space


with open("../data/wsdm/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

for data in tqdm(train_data):
    data["question"] = deal_text(data["question"])
    data["answer"] = deal_text(data["answer"])
    document_list = []
    for i, document in enumerate(data["documents"]):
        if len(document) == 0:
            continue
        document_list.append(deal_text(document))

    data["documents"] = document_list

    history_list = []
    for i, q_and_a in enumerate(data["history"]):
        temp_history_dict = {}
        temp_history_dict["question"] = deal_text(q_and_a["question"])
        temp_history_dict["answer"] = deal_text(q_and_a["answer"])
        history_list.append(temp_history_dict)

    data["history"] = history_list


with open("../data/wsdm/clean/release_train_data.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)


with open("../data/wsdm/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
    eval_data = json.load(f)

for data in tqdm(eval_data):
    data["question"] = deal_text(data["question"])
    document_list = []
    for i, document in enumerate(data["documents"]):
        if len(document) == 0:
            continue
        document_list.append(deal_text(document))

    data["documents"] = document_list

    history_list = []
    for i, q_and_a in enumerate(data["history"]):
        temp_history_dict = {}
        temp_history_dict["question"] = deal_text(q_and_a["question"])
        temp_history_dict["answer"] = deal_text(q_and_a["answer"])
        history_list.append(temp_history_dict)

    data["history"] = history_list


with open("../data/wsdm/clean/release_phase1_eval_data_wo_gt.json", "w") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=4)

# print(len(list(set(emoji_list_drop))))
# print(sorted(list(set(emoji_list_drop))))
# print(len(list(set(emoji_list_reserved))))
# print(sorted(list(set(emoji_list_reserved))))
