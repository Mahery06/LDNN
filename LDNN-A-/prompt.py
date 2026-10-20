general_prompt=(
                "现在问题的具体背景是{type_in}."
                "具体如何选择节点的方式请参考{data}."
                "Your task: "
                "1.依据state([[TempPath],'content of thought',[TempPath],'content of thought',……])来选择下一步可能的节点;"
                "2.你需要依据{input}来生成4个可能的节点."
                "My Request:"
                # "1.选取节点与原来节点构成的张量(array([chosen_point_x-former_point_x,chosen_point_y-former_point_y])一定要与给定张量{ts}的夹角余弦值落在0~1的范围内;"
                "1.路径点的选取贴着障碍物选取，不要刻意远离障碍物。"
                "2.barriers不是非要避开的，若barriers之间的空隙可以走，也是可以走的。"
                "3.每个候选节点与路径上所有已有点，至少在 x 或 y 方向上相差 ≥2 格（不要生成与已有点相邻的格子）。)"
                "4.Thought 内容严格以[{{\"Point\":[x,y],\"Thought\":\"content of thought\"}},{{\"Point\":[x,y],\"Thought\":\"content of thought\"}}……}}]格式输出,','一定要写."
                "并以str输出。")

