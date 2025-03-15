from typing import IO, Optional

from dify_plugin import Speech2TextModel
import httpx
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeBadRequestError, InvokeError

import assemblyai as aai

class AssemblyAISpeech2TextModel(Speech2TextModel):
    """
    Model class for OpenAI Speech to text model.
    """
    def _invoke(self, model: str, credentials: dict,
                file: IO[bytes], user: Optional[str] = None) \
            -> str:
        """
        Invoke speech2text model

        :param model: model name
        :param credentials: model credentials
        :param file: audio file
        :param user: unique user id
        :return: text for given audio file
        """
        return self._speech2text_invoke(model, credentials, file)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            audio_file_path = self._get_demo_file_path()

            with open(audio_file_path, 'rb') as audio_file:
                self._speech2text_invoke(model, credentials, audio_file)
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _speech2text_invoke(self, model: str, credentials: dict, file: IO[bytes]) -> str:
        """
        Invoke speech2text model

        :param model: model name
        :param credentials: model credentials
        :param file: audio file
        :return: text for given audio file
        """
        api_key = credentials.get("api_key")
        aai.settings.api_key = api_key
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best if model == "best" else aai.SpeechModel.nano,
            language_code=aai.LanguageCode.zh,
        )

        client = aai.Transcriber(config=config)
        transcript = client.transcribe(file.read())

        if transcript.status == aai.TranscriptStatus.error:
            raise InvokeBadRequestError(transcript.error)

        return transcript.text

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeBadRequestError: [
                httpx.HTTPStatusError,
            ],
        }
