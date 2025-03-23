"""LLMクライアントのファクトリーモジュール。"""

import os
import logging
from typing import Optional, Union

from dotenv import load_dotenv

from .grok_client import Grok3Client
from .litellm_client import LiteLLMClient

# ロガーの設定
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()


class LLMClientFactory:
    """LLMクライアントのファクトリークラス。"""
    
    @staticmethod
    def create_client(
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ) -> Union[Grok3Client, LiteLLMClient]:
        # 環境変数から設定を読み込む
        provider = provider or os.getenv("LLM_PROVIDER", "litellm")
        logger.info(f"Using LLM provider: {provider}")
        """
        LLMクライアントを作成します。
        
        Parameters
        ----------
        provider : str, optional
            LLMプロバイダー。指定しない場合は環境変数から取得。
        api_key : str, optional
            APIキー。指定しない場合は環境変数から取得。
        base_url : str, optional
            APIのベースURL。指定しない場合は環境変数から取得。
        model : str, optional
            使用するモデル名。指定しない場合は環境変数から取得。
            
        Returns
        -------
        Union[Grok3Client, LiteLLMClient]
            LLMクライアントのインスタンス。
            
        Raises
        ------
        ValueError
            不正なプロバイダーが指定された場合。
        """
        provider = provider.lower()
        if provider == "grok":
            logger.info("Creating Grok client")
            return Grok3Client(api_key=api_key)
        elif provider == "litellm":
            logger.info("Creating LiteLLM client")
            logger.info(f"LiteLLM base URL: {base_url or os.getenv('LITELLM_API_BASE', 'http://localhost:4000')}")
            logger.info(f"LiteLLM model: {model or os.getenv('LITELLM_MODEL', 'bedrock-converse-us-claude-3-7-sonnet-v1')}")
            return LiteLLMClient(
                api_key=api_key,
                base_url=base_url,
                model=model
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None
) -> Union[Grok3Client, LiteLLMClient]:
    """
    LLMクライアントのインスタンスを取得します。
    
    Parameters
    ----------
    provider : str, optional
        LLMプロバイダー。指定しない場合は環境変数から取得。
    api_key : str, optional
        APIキー。指定しない場合は環境変数から取得。
    base_url : str, optional
        APIのベースURL。指定しない場合は環境変数から取得。
    model : str, optional
        使用するモデル名。指定しない場合は環境変数から取得。
        
    Returns
    -------
    Union[Grok3Client, LiteLLMClient]
        LLMクライアントのインスタンス。
    """
    return LLMClientFactory.create_client(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model
    )
