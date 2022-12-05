#! python3.7

# Needed to cx_Freeze as a GUI app. https://stackoverflow.com/a/3237924
import sys
class dummyStream:
    ''' dummyStream behaves like a stream but does nothing. '''
    def __init__(self): pass
    def write(self,data): pass
    def read(self,data): pass
    def flush(self): pass
    def close(self): pass
# Redirect all default streams to this dummyStream:
sys.stdout = dummyStream()
sys.stderr = dummyStream()
sys.stdin = dummyStream()
sys.__stdout__ = dummyStream()
sys.__stderr__ = dummyStream()
sys.__stdin__ = dummyStream()


import audioop
import flet as ft
import io
import os
import numpy
import pyaudio
import wave
import whisper
import yaml

from datetime import datetime, timedelta
from queue import Queue
from threading import Thread
from time import sleep
from whisper.tokenizer import LANGUAGES


def main(page: ft.Page):
    #
    # Settings and constants.
    #

    # Used to store transcription when done.
    settings_file = "transcriber_settings.yaml"
    settings = {}
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f)
    # Happens if the file is empty.
    if settings == None:
        settings = {}

    transcription_file = "transcription.txt"
    max_energy = 5000
    sample_rate = 16000
    chunk_size = 1024
    max_int16 = 2**15

    # Set window settings.
    page.title = "Transcriber"
    page.window_min_width = 817.0
    page.window_width = settings.get('window_width', page.window_min_width)
    page.window_min_height = 475.0
    page.window_height = settings.get('window_height', 800.0)

    #
    # Callbacks.
    #

    def always_on_top_callback(_):
       page.window_always_on_top = always_on_top_checkbox.value
       page.update()

    def text_background_callback(_):
        if text_background_checkbox.value:
            for list_item in transcription_list.controls:
                list_item.bgcolor = ft.colors.BLACK if dark_mode_checkbox.value else ft.colors.WHITE
        else:
            for list_item in transcription_list.controls:
                list_item.bgcolor = None
        transcription_list.update()

    def dark_mode_callback(_):
        if dark_mode_checkbox.value:
            page.theme_mode = ft.ThemeMode.DARK
        else:
            page.theme_mode = ft.ThemeMode.LIGHT
        text_background_callback(_)
        page.update()

    def language_callback(_):
        translate_checkbox.disabled = language_dropdown.value == 'en'
        if language_dropdown.value == 'en':
            translate_checkbox.value = False
        translate_checkbox.update()

    def text_size_callback(_):
        for list_item in transcription_list.controls:
            list_item.size = int(text_size_dropdown.value)
        transcription_list.update()

    audio_model:whisper.Whisper = None
    loaded_audio_model:str = None
    currently_transcribing = False
    stop_recording:function = None
    record_thread:Thread = None
    data_queue = Queue()
    def transcribe_callback(_):
        nonlocal currently_transcribing, audio_model, stop_recording, loaded_audio_model, record_thread, run_record_thread
        if not currently_transcribing:
            page.splash = ft.Container(
                content=ft.ProgressRing(),
                alignment=ft.alignment.center
            )
            page.update()

            model = model_dropdown.value
            if model != "large" and language_dropdown.value == 'en':
                model = model + ".en"

            # Only re-load the audio model if it changed.
            if (not audio_model or not loaded_audio_model) or ((audio_model and loaded_audio_model) and loaded_audio_model != model):
                audio_model = whisper.load_model(model)
                loaded_audio_model = model

            device_index = int(microphone_dropdown.value)
            if not record_thread:
                stream = pa.open(format=pyaudio.paInt16,
                                 channels=1,
                                 rate=sample_rate,
                                 input=True,
                                 frames_per_buffer=chunk_size,
                                 input_device_index=device_index)
                record_thread = Thread(target=recording_thread, args=[stream])
                run_record_thread = True
                record_thread.start()

            transcribe_text.value = "Stop Transcribing"
            transcribe_icon.name = "stop_rounded"
            transcribe_button.bgcolor = ft.colors.RED_800

            # Disable all the controls.
            model_dropdown.disabled = True
            microphone_dropdown.disabled = True
            language_dropdown.disabled = True
            translate_checkbox.disabled = True
            settings_controls.visible = False

            # Make transparent.
            if transparent_checkbox.value:
                page.window_bgcolor = ft.colors.TRANSPARENT
                page.bgcolor = ft.colors.TRANSPARENT
                page.window_title_bar_hidden = True
                page.window_frameless = True
                draggable_area1.visible = True
                draggable_area2.visible = True

            # Save all settings.
            settings = {
                'window_width': page.window_width,
                'window_height': page.window_height,
                'speech_model': model_dropdown.value,
                'microphone_index': microphone_dropdown.value,
                'language': language_dropdown.value,
                'text_size': text_size_dropdown.value,
                'translate': translate_checkbox.value,
                'always_on_top': always_on_top_checkbox.value,
                'dark_mode': dark_mode_checkbox.value,
                'text_background': text_background_checkbox.value,
                'transparent': transparent_checkbox.value,
                'volume_threshold': energy_slider.value,
                'transcribe_rate': transcribe_rate_seconds,
                'max_record_time': max_record_time,
                'seconds_of_silence_between_lines': silence_time,
            }

            with open(settings_file, 'w+') as f:
                yaml.dump(settings, f)

            currently_transcribing = True
        else:
            page.splash = ft.Container(
                content=ft.ProgressRing(),
                alignment=ft.alignment.center
            )
            page.update()

            transcribe_text.value = "Start Transcribing"
            transcribe_icon.name = "play_arrow_rounded"
            transcribe_button.bgcolor = ft.colors.BLUE_800
            volume_bar.value = 0.01

            # Stop the record thread.
            if record_thread:
                run_record_thread = False
                record_thread.join()
                record_thread = None

            # Drain all the remaining data but save the last sample.
            # This is to pump the main loop one more time, otherwise we'll end up editing
            # the last line when we start transcribing again, rather than creating a new line.
            data = None
            while not data_queue.empty():
                data = data_queue.get()
            if data:
                data_queue.put(data)

            # Enable all the controls.
            model_dropdown.disabled = False
            microphone_dropdown.disabled = False
            language_dropdown.disabled = False
            translate_checkbox.disabled = language_dropdown.value == 'en'
            settings_controls.visible = True

            # Make opaque.
            page.window_bgcolor = None
            page.bgcolor = None
            page.window_title_bar_hidden = False
            page.window_frameless = False
            draggable_area1.visible = False
            draggable_area2.visible = False

            # Save transcription.
            with open(transcription_file, 'w+', encoding='utf-8') as f:
                f.writelines('\n'.join([item.value for item in transcription_list.controls]))

            currently_transcribing = False

        page.splash = None
        page.update()


    #
    # Build controls.
    #

    model_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option('tiny', text="Tiny (Fastest)"),
            ft.dropdown.Option('base', text="Base"),
            ft.dropdown.Option('small', text="Small"),
            ft.dropdown.Option('medium', text="Medium"),
            ft.dropdown.Option('large', text="Large (Highest Quality)"),
        ],
        label="Speech To Text Model",
        value=settings.get('speech_model', 'base'),
        expand=True,
        content_padding=ft.padding.only(top=5, bottom=5, left=10),
        text_size=14,
    )

    microphones = {}
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        device_info = pa.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0 and device_info['hostApi'] == 0:
            microphones[device_info['index']] = device_info['name']

    default_mic = pa.get_default_input_device_info()['index']
    selected_mic = int(settings.get('microphone_index', default_mic))
    if selected_mic not in microphones:
        selected_mic = default_mic

    microphone_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option(index, text=mic) for index, mic in microphones.items()],
        label="Audio Input Device",
        value=selected_mic,
        expand=True,
        content_padding=ft.padding.only(top=5, bottom=5, left=10),
        text_size=14,
    )

    language_options = [ft.dropdown.Option("Auto")]
    language_options += [ft.dropdown.Option(abbr, text=lang.capitalize()) for abbr, lang in LANGUAGES.items()]
    language_dropdown = ft.Dropdown(
        options=language_options,
        label="Language",
        value=settings.get('language', "Auto"),
        content_padding=ft.padding.only(top=5, bottom=5, left=10),
        text_size=14,
        on_change=language_callback
    )

    text_size_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option(size) for size in range(8, 66, 2)],
        label="Text Size",
        value=settings.get('text_size', 24),
        on_change=text_size_callback,
        content_padding=ft.padding.only(top=5, bottom=5, left=10),
        text_size=14,
    )

    translate_checkbox = ft.Checkbox(label="Translate To English", value=settings.get('translate', False), disabled=language_dropdown.value == 'en')
    dark_mode_checkbox = ft.Checkbox(label="Dark Mode", value=settings.get('dark_mode', False), on_change=dark_mode_callback)
    text_background_checkbox = ft.Checkbox(label="Text Background", value=settings.get('text_background', False), on_change=text_background_callback)
    always_on_top_checkbox = ft.Checkbox(label="Always On Top", value=settings.get('always_on_top', False), on_change=always_on_top_callback)
    transparent_checkbox = ft.Checkbox(label="Transparent", value=settings.get('transparent', False))

    energy_slider = ft.Slider(min=0, max=max_energy, value=settings.get('volume_threshold', 300), expand=True, height=20)
    volume_bar = ft.ProgressBar(value=0.01, color=ft.colors.RED_800)

    transcription_list = ft.ListView([], spacing=10, padding=20, expand=True, auto_scroll=True)

    transcribe_text = ft.Text("Start Transcribing")
    transcribe_icon = ft.Icon("play_arrow_rounded")

    transcribe_button = ft.ElevatedButton(
        content=ft.Row(
            [
                transcribe_icon,
                transcribe_text
            ],
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=5
        ),
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=5)),
        bgcolor=ft.colors.BLUE_800, color=ft.colors.WHITE,
        on_click=transcribe_callback,
    )

    settings_controls = ft.Column(
        [
            ft.Container(
                content=ft.Row(
                    [
                        model_dropdown,
                        ft.Icon("help_outline", tooltip="Choose which model to transcribe speech with.\nModels are downloaded automatically the first time they are used.")
                    ],
                    spacing=10,
                ),
                padding=ft.padding.only(left=10, right=10, top=15),
            ),
            ft.Container(
                content=microphone_dropdown,
                padding=ft.padding.only(left=10, right=45, top=5)
            ),
            ft.Container(
                content=ft.Row(
                    [
                        ft.Column(
                            [
                                language_dropdown,
                                translate_checkbox,
                            ]
                        ),
                        ft.Container(
                            content=ft.Column(
                                [
                                    text_size_dropdown,
                                    ft.Row(
                                        [
                                            text_background_checkbox,
                                            dark_mode_checkbox,
                                        ]
                                    ),
                                ],
                            ),
                            padding=ft.padding.only(left=10)
                        ),
                        ft.Column(
                            [
                                ft.Row(
                                    [
                                        transparent_checkbox,
                                        ft.Icon("help_outline", tooltip="Make the window transparent while transcribing.")
                                    ]
                                ),
                                always_on_top_checkbox,
                            ]
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                margin=ft.margin.only(left=10, right=15, top=5),
            ),
            ft.Container(
                content=ft.Row(
                    [
                        energy_slider,
                        ft.Icon("help_outline", tooltip="Required volume to start decoding speech.\nAdjusts max volume automatically.")
                    ],
                    expand=True,
                ),
                padding=ft.padding.only(left=0, right=15, top=0),
            ),
        ],
        visible=True
    )

    draggable_area1 = ft.Row(
        [
            ft.WindowDragArea(ft.Container(height=30), expand=True),
        ],
        visible=False
    )
    draggable_area2 = ft.Row(
        [
            ft.WindowDragArea(ft.Container(height=30), expand=True),
        ],
        visible=False
    )

    page.add(
        settings_controls,
        draggable_area1,
        ft.Container(
            content=transcribe_button,
            padding=ft.padding.only(left=10, right=45, top=5)
        ),
        ft.Container(
            content=volume_bar,
            padding=ft.padding.only(left=10, right=45, top=0)
        ),
        draggable_area2,
        ft.Container(
            content=transcription_list,
            padding=ft.padding.only(left=15, right=45, top=5),
            expand=True,
        ),
    )

    # Set settings that may have been loaded.
    dark_mode_callback(None)
    always_on_top_callback(None)
    text_background_callback(None)

    #
    # Control loops.
    #

    run_record_thread = True
    def recording_thread(stream:pyaudio.Stream):
        nonlocal max_energy
        while run_record_thread:
            # We record as fast as possible so that we can update the volume bar at a fast rate.
            data = stream.read(chunk_size)
            energy = audioop.rms(data, pa.get_sample_size(pyaudio.paInt16))
            if energy > max_energy:
                max_energy = energy
                energy_slider.max = max_energy
                energy_slider.update()
            volume_bar.value = min(energy / max_energy, 1.0)
            if energy < energy_slider.value:
                volume_bar.color = ft.colors.RED_800
            else:
                volume_bar.color = ft.colors.BLUE_800
            data_queue.put(data)
            volume_bar.update()

    next_transcribe_time = None
    transcribe_rate_seconds = float(settings.get('transcribe_rate', 0.5))
    transcribe_rate = timedelta(seconds=transcribe_rate_seconds)
    max_record_time = settings.get('max_record_time', 30)
    silence_time = settings.get('seconds_of_silence_between_lines', 0.5)
    last_sample = bytes()
    samples_with_silence = 0
    while True:
        # Main loop. Wait for data from the recording thread at transcribe it at specified rate.
        if currently_transcribing and audio_model and not data_queue.empty():
            now = datetime.utcnow()
            # Set next_transcribe_time for the first time.
            if not next_transcribe_time:
                next_transcribe_time = now + transcribe_rate

            # Only run transcription occasionally. This reduces stress on the GPU and makes transcriptions
            # more accurate because they have more audio context, but makes the transcription less real time.
            if now > next_transcribe_time:
                next_transcribe_time = now + transcribe_rate

                phrase_complete = False
                while not data_queue.empty():
                    data = data_queue.get()
                    energy = audioop.rms(data, pa.get_sample_size(pyaudio.paInt16))
                    if (energy < energy_slider.value):
                        samples_with_silence += 1
                    else:
                        samples_with_silence = 0

                    # If we have encounter enough silence, restart the buffer and add a new line.
                    if samples_with_silence > sample_rate / chunk_size * silence_time:
                        phrase_complete = True
                        last_sample = bytes()
                    last_sample += data

                # Write out raw frames as a wave file.
                wav_file = io.BytesIO()
                wav_writer:wave.Wave_write = wave.open(wav_file, "wb")
                wav_writer.setframerate(sample_rate)
                wav_writer.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                wav_writer.setnchannels(1)
                wav_writer.writeframes(last_sample)
                wav_writer.close()

                # Read the audio data, now with wave headers.
                wav_file.seek(0)
                wav_reader:wave.Wave_read = wave.open(wav_file)
                samples = wav_reader.getnframes()
                audio = wav_reader.readframes(samples)
                wav_reader.close()

                # Convert the wave data straight to a numpy array for the model.
                # https://stackoverflow.com/a/62298670
                audio_as_np_int16 = numpy.frombuffer(audio, dtype=numpy.int16)
                audio_as_np_float32 = audio_as_np_int16.astype(numpy.float32)
                audio_normalised = audio_as_np_float32 / max_int16

                language = None
                if language_dropdown.value != 'Auto':
                    language = language_dropdown.value

                task = 'transcribe'
                if language != 'en' and translate_checkbox.value:
                    task = 'translate'

                result = audio_model.transcribe(audio_normalised, language=language, task=task)
                text = result['text'].strip()

                color = None
                if text_background_checkbox.value:
                    color = ft.colors.BLACK if dark_mode_checkbox.value else ft.colors.WHITE

                if not phrase_complete and transcription_list.controls:
                    transcription_list.controls[-1].value = text
                elif not transcription_list.controls or (transcription_list.controls and transcription_list.controls[-1].value):
                    # Always add a new item if there are no items in the list.
                    # Only add another item to the list if the previous item is not an empty string.
                    # Since hearing silence triggers phrase_complete, there's a good chance that most appends are going to empty text.
                    transcription_list.controls.append(ft.Text(text, selectable=True, size=int(text_size_dropdown.value), bgcolor=color))
                transcription_list.update()

                # If we've reached our max recording time, it's time to break up the buffer, add an empty line after we edited the last line.
                audio_length_in_seconds = samples / float(sample_rate)
                if audio_length_in_seconds > max_record_time:
                    last_sample = bytes()
                    transcription_list.controls.append(ft.Text('', selectable=True, size=int(text_size_dropdown.value), bgcolor=color))

        sleep(0.1)


if __name__ == "__main__":
    ft.app(target=main)
