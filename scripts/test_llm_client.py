#!/usr/bin/env python
"""LLMクライアントのテストスクリプト。"""

import os
import sys
import logging
from dotenv import load_dotenv

# nookパッケージをインポートできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nook.common.llm_factory import get_llm_client

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()


def main():
    """メイン関数。"""
    # 環境変数の確認
    provider = os.environ.get("LLM_PROVIDER")
    api_base = os.environ.get("LITELLM_API_BASE")
    api_key = os.environ.get("LITELLM_API_KEY")
    model = os.environ.get("LITELLM_MODEL")
    
    logger.info("環境変数の設定:")
    logger.info(f"LLM_PROVIDER: {provider}")
    logger.info(f"LITELLM_API_BASE: {api_base}")
    logger.info(f"LITELLM_API_KEY: {'設定済み' if api_key else '未設定'}")
    logger.info(f"LITELLM_MODEL: {model}")
    
    try:
        # LLMクライアントを取得
        llm_client = get_llm_client()
        logger.info(f"クライアントの種類: {type(llm_client).__name__}")
        
        # テキスト生成のテスト
        prompt = "こんにちは！今日の天気はどうですか？"
        logger.info(f"プロンプト: {prompt}")
        
        response = llm_client.generate_content(prompt)
        logger.info(f"応答: {response}")
        
        # チャットのテスト
        logger.info("チャットセッションのテスト:")
        chat_session = llm_client.create_chat(
            system_instruction="あなたは役立つアシスタントです。簡潔に回答してください。"
        )
        
        messages = [
            "自己紹介をお願いします。",
            "あなたの得意なことは何ですか？",
            "ありがとう！"
        ]
        
        for message in messages:
            logger.info(f"ユーザー: {message}")
            response = llm_client.send_message(chat_session, message)
            logger.info(f"アシスタント: {response}")
            
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
