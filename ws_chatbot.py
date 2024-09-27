#import socket, threading, datetime, ast,json, torch
print("Starting script")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, ssl
print("everything imported")
device = "cuda:0" if torch.cuda.is_available else "cpu"
print(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)
print("loaded models")
import asyncio, websockets
#from gtts import gTTS
from io import BytesIO

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
data_dcit = {"jason":"Jason is out Technical and Clinical lead. Having worked in the NHS for 15 years, Jason transitioned into MedTech and AI to try to help as many people as possible. Jason is passionate about Mental Health and the link between mental health and artists and their works.",
	"michael":"Michael is an avid traveler and historian. Having experience in teaching and research, Artificial Intelligence and Simulation   is a direct fit for his skill set.",
	"goals":"Carrier Technologies is passionate about creating tools to advance Mental Health. Mental Health is sadly a very subjective field, it is our ultimate aim to create an objective measurement system to advance the field. To do so will take a Herculean effort, but it is needed. By using AI to help as much as it can, gathering data in a transparent and open nature, Carrier Technologies aims to create a highly engageable platform to encouragte dialogue and research.",
	"about":"Carrier Technologies is a small start up focused on mental health, AI and their interactions",
        "whatissoundboard":"SoundBoard is a mental wellness/health app that aims at exploring users emotions. Emotions are complex and multidimensional and yet in speech we polarise them down to a few possibilites (happy, sad, angry, etc) and even those definitions are not objective, if someone says they are  upset it might mean angry, it might mean sad. By using prosody and a gentle LLM, SoundBoard aims to look deeper into what emotions we really have."}

async def handle_conn(websocket, path):
    while True:
    	data = await websocket.recv()
    	print(data)
    	l = [w.lower() for w in data.split(" ") if w.lower() in data_dcit]
    	if l:
            await websocket.send(data_dcit[l[0]])
    	elif "Jane asks for tts" in data:
        	print("got tts request")
        	mp3 = BytesIO()
        	tts = gTTS(data)
        	tts.write_to_fp(mp3)
        	mp3.seek(0)
        	#print(f"should be sending the file now of length {len(mp3.read())}")
        	await websocket.send(mp3.read())
    	else:
        	new_user_input_ids = tokenizer.encode(data + tokenizer.eos_token, return_tensors='pt').to(device)
        	bot_input_ids = new_user_input_ids#torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        	chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        	tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        	await websocket.send(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(
        "/etc/letsencrypt/live/carriertech.uk/fullchain.pem",
        "/etc/letsencrypt/live/carriertech.uk/privkey.pem")
context.load_verify_locations(cafile='/var/www/html/ca.crt')
start_server = websockets.serve(handle_conn, port=8774, ssl=context)
print("starting server on 8774")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
