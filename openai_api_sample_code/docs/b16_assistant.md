### Assistant　説明
| object    | What it represents: それが表すもの                                                                                    |
|-----------|----------------------------------------------------------------------------------------------------------------|
| Assistant | OpenAIのモデルを使用し、ツールを呼び出す専用のAI                                                                                   |
| Thread    | アシスタントとユーザー間の会話セッション。スレッドはメッセージを保存し、コンテンツをモデルのコンテキストに合わせるために自動的に切り捨てを処理します。                                    |
| Message	  | アシスタントまたはユーザーによって作成されたメッセージ。メッセージには、テキスト、画像、その他のファイルを含めることができます。メッセージはスレッドにリストとして保存されます。                       |
| Run       | スレッドでのアシスタントの呼び出し。アシスタントは、その構成とスレッドのメッセージを使用して、モデルとツールを呼び出してタスクを実行します。実行の一部として、アシスタントはスレッドにメッセージを追加します。        |
| Run Step	 | アシスタントが実行の一環として実行した手順の詳細なリスト。アシスタントは実行中にツールを呼び出したり、メッセージを作成したりできます。実行手順を調べると、アシスタントが最終結果にどのように到達しているかを詳しく調べることができます。 |

### assistant tools種別
| 題目              | 説明                                                                                                                        |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------|
| アシスタントの作成 | アシスタントを作成するには、カスタム指示を定義し、モデルを選択します。必要に応じて、ファイルを追加し、Code Interpreter、File Search、Function Callingなどのツールを有効にします。 |
| スレッドの作成     | ユーザーが会話を開始するときにスレッドを作成します。                                                                          |
| メッセージの追加   | ユーザーが質問をするたびに、スレッドにメッセージを追加します。                                                                    |
| アシスタントの実行 | スレッド上でアシスタントを実行し、モデルとツールを呼び出して応答を生成します。                                                   |

