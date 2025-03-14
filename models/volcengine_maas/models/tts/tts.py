import asyncio
import copy
import gzip
import json
import uuid
import websockets
from collections.abc import Generator
from io import BytesIO
from typing import Optional

from dify_plugin import TTSModel
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from legacy.errors import (
    AuthErrors,
    BadRequestErrors,
    ConnectionErrors,
    RateLimitErrors,
    ServerUnavailableErrors,
)

MESSAGE_TYPES = {
    11: "audio-only server response",
    12: "frontend server response",
    15: "error message from server"
}
MESSAGE_TYPE_SPECIFIC_FLAGS = {
    0: "no sequence number",
    1: "sequence number > 0",
    2: "last message from server (seq < 0)",
    3: "sequence number < 0"
}
MESSAGE_SERIALIZATION_METHODS = {0: "no serialization", 1: "JSON", 15: "custom type"}
MESSAGE_COMPRESSIONS = {0: "no compression", 1: "gzip", 15: "custom compression method"}

class VolcengineMaaSTTSModel(TTSModel):
    """
    Model class for Volcengine Maas Speech to text model.
    """
    # version: b0001 (4 bits)
    # header size: b0001 (4 bits)
    # message type: b0001 (Full client request) (4bits)
    # message type specific flags: b0000 (none) (4bits)
    # message serialization method: b0001 (JSON) (4bits)
    # message compression: b0001 (gzip) (4bits)
    # reserved data: 0x00 (1 byte)
    DEFAULT_HEADER = bytearray(b'\x11\x10\x11\x00')

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> bytes | Generator[bytes, None, None]:
        """
        Invoke text2speech model

        :param model: model name
        :param tenant_id: user tenant id
        :param credentials: model credentials
        :param content_text: text content to be translated
        :param voice: model timbre
        :param user: unique user id
        :return: text translated to audio file or generator of audio chunks
        """
        if not voice or voice not in [
            d["value"] for d in self.get_tts_model_voices(model=model, credentials=credentials)
        ]:
            voice = self._get_model_default_voice(model, credentials)
        return self._invoke_streaming(
            model=model,
            credentials=credentials, 
            content_text=content_text, 
            voice=voice,
            user=user
        )

    def validate_credentials(
        self, model: str, credentials: dict, user: Optional[str] = None
    ) -> None:
        """
        Validate credentials for text2speech model

        :param model: model name
        :param credentials: model credentials
        :param user: unique user id
        """
        try:
            self._invoke(
                model=model,
                credentials=credentials,
                content_text="Hello Dify!",
                voice=self._get_model_default_voice(model, credentials),
                user=user,
                tenant_id="",
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _invoke_streaming(
        self,
        model: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> Generator[bytes, None, None]:
        """
        Streaming version of text2speech model invocation

        :param model: model name
        :param credentials: model credentials
        :param content_text: text content to be translated
        :param voice: model timbre
        :param user: unique user id
        :return: generator of audio chunks
        """
        word_limit = self._get_model_word_limit(model, credentials)
        audio_type = self._get_model_audio_type(model, credentials)

        # Split text into sentences
        sentences = self._split_text_into_sentences(
            content_text, max_length=word_limit
        )

        base_request = {
            "app": {
                "appid": credentials.get("endpoint_id"),
                "token": credentials.get("volc_access_key_id"),
                "cluster": credentials.get("volc_region")
            },
            "user": {
                "uid": user or str(uuid.uuid4())
            },
            "audio": {
                "voice_type": voice,
                "encoding": audio_type,
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            }
        }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            for sentence in sentences:
                request = copy.deepcopy(base_request)
                request["request"] = {
                    "reqid": str(uuid.uuid4()),
                    "text": sentence,
                    "text_type": "plain",
                    "operation": "submit"
                }
                
                chunk = loop.run_until_complete(self._get_audio_data(request, credentials))
                yield chunk
        finally:
            loop.close()

    async def _get_audio_data(self, request_json: dict, credentials: dict) -> bytes:
        """
        Get audio data using websocket connection

        :param request_json: request payload
        :param credentials: model credentials
        :return: audio bytes
        """
        host = credentials.get("api_endpoint_host", "wss://openspeech.bytedance.com")
        api_url = f"{host}/api/v1/tts/ws_binary"

        # Prepare request payload
        payload_bytes = str.encode(json.dumps(request_json))
        payload_bytes = gzip.compress(payload_bytes)
        
        # Create full request
        full_client_request = bytearray(self.DEFAULT_HEADER)
        full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        full_client_request.extend(payload_bytes)

        # Prepare headers
        headers = {"Authorization": f"Bearer; {credentials['volc_access_key_id']}"}
        audio_buffer = BytesIO()

        async with websockets.connect(api_url, extra_headers=headers, ping_interval=None) as ws:
            await ws.send(full_client_request)
            
            while True:
                response = await ws.recv()
                done = self._parse_response(response, audio_buffer)
                if done:
                    break

        return audio_buffer.getvalue()

    def _parse_response(self, response: bytes, audio_buffer: BytesIO) -> bool:
        """
        Parse websocket response and write audio data to buffer

        :param response: websocket response bytes
        :param audio_buffer: buffer to write audio data
        :return: True if parsing is complete, False if more data expected
        """
        # Parse header
        message_type = response[1] >> 4
        message_type_specific_flags = response[1] & 0x0f
        message_compression = response[2] & 0x0f
        header_size = response[0] & 0x0f
        payload = response[header_size * 4:]

        # Handle audio response
        if message_type == 0xb:  # audio-only server response
            if message_type_specific_flags == 0:  # no sequence number as ACK
                return False
            
            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            audio_data = payload[8:]
            
            audio_buffer.write(audio_data)
            return sequence_number < 0

        # Handle error response
        elif message_type == 0xf:
            code = int.from_bytes(payload[:4], "big", signed=False)
            msg_size = int.from_bytes(payload[4:8], "big", signed=False)
            error_msg = payload[8:]
            
            if message_compression == 1:
                error_msg = gzip.decompress(error_msg)
            error_msg = str(error_msg, "utf-8")
            
            raise InvokeBadRequestError(f"Error code {code}: {error_msg}")

        # Handle frontend response
        elif message_type == 0xc:
            msg_size = int.from_bytes(payload[:4], "big", signed=False)
            msg_payload = payload[4:]
            if message_compression == 1:
                msg_payload = gzip.decompress(msg_payload)
            return True

        return False

    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: ConnectionErrors.values(),
            InvokeServerUnavailableError: ServerUnavailableErrors.values(),
            InvokeRateLimitError: RateLimitErrors.values(),
            InvokeAuthorizationError: AuthErrors.values(),
            InvokeBadRequestError: BadRequestErrors.values(),
        }
